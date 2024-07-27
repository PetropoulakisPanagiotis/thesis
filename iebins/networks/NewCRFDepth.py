import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer import SwinTransformer
from .newcrf_layers import NewCRF
from .uper_crf_head import PSP
from .depth_update import *
from .utils import *


class NewCRFDepth(nn.Module):
    """
    Depth network based on neural window FC-CRFs architecture
    """
    def __init__(self, version=None, pretrained=None, min_depth=0.1, max_depth=100.0, max_tree_depth=1,
                    update_block=0, loss_type=0, train_decoder=0, \
                    num_semantic_classes=14, num_instances=63, padding_instances=0, \
                    segmentation_active=False, concat_masks=False, instances_active=False, roi_align=False, roi_align_size=32, \
                    unc_head=False, virtual_depth_variation=0, upsample_type=0, bins_type=1, \
                    bin_num=16, bins_scale=50, bins_type_scale=1, **kwargs):
        super().__init__()

        self.freeze_backbone = True
        self.train_decoder = train_decoder  # train the last layer of the decoder

        # IEBINS, global, per class or per instance scale #
        self.update_block = update_block

        ##############################
        # Bins                       #
        # 0 bins scale/canonical     #
        # 1 bins canonical           #
        # 2 bins scale               #
        # regression scale/canonical #
        ##############################
        self.virtual_depth_variation = virtual_depth_variation

        self.max_tree_depth = max_tree_depth
        self.bin_num = bin_num
        self.bins_type = bins_type
        self.min_depth = min_depth
        self.max_depth = max_depth  # metric or canonical
        self.bins_scale = bins_scale
        self.bins_type_scale = bins_type_scale

        # Segmentation #
        self.num_semantic_classes = num_semantic_classes
        self.segmentation_active = segmentation_active
        self.concat_masks = concat_masks

        # Instances #
        self.instances_active = instances_active
        self.num_instances = num_instances
        self.padding_instances = padding_instances
        self.roi_align = roi_align
        self.roi_align_size = roi_align_size

        self.unc_head = unc_head  # Uncertainty prediction via additional heads

        #############################################################
        # 0: Torch, 1: custom bilinear                              #
        # 2: for uncertainty weights^2 --> see Rosinol et. al. 2023 #
        #############################################################
        self.upsample_type = upsample_type
        self.loss_type = loss_type

        # Print some info to console #
        self.network_info_print()

        # Backbone dims #
        if version[:-2] == 'base':
            embed_dim = 128
            depths = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]
            in_channels = [128, 256, 512, 1024]
        elif version[:-2] == 'large':
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
            in_channels = [192, 384, 768, 1536]
        elif version[:-2] == 'tiny':
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            in_channels = [96, 192, 384, 768]

        self.hidden_dim = 128
        self.context_dim = 96

        ##############################################
        # Set update block for depth prediction head #
        ##############################################

        ##########
        # IEBINS #
        ##########
        if self.update_block == 0:
            self.update = IEBINS(hidden_dim=self.hidden_dim, context_dim=self.context_dim, bin_num=bin_num,
                                 upsample_type=self.upsample_type)
            print("[VARIATION IEBINS]\n")

        ################
        # Global scale #
        ################
        elif self.update_block == 1:
            self.update = GlobalScale(hidden_dim=self.hidden_dim, context_dim=self.context_dim, bin_num=self.bin_num, loss_type=self.loss_type, bins_scale=bins_scale, \
                                      virtual_depth_variation=self.virtual_depth_variation, upsample_type=self.upsample_type, \
                                      bins_type=self.bins_type, bins_type_scale=self.bins_type_scale)
            print("[VARIATION GlobalScale]\n")

        ###################
        # Per-Class scale #
        ###################
        elif self.update_block == 2:
            self.update = PerClassScale(hidden_dim=self.hidden_dim, context_dim=self.context_dim, bin_num=self.bin_num, \
                                                                   loss_type=self.loss_type, num_semantic_classes=self.num_semantic_classes, bins_scale=bins_scale, \
                                                                   virtual_depth_variation=self.virtual_depth_variation, upsample_type=self.upsample_type, bins_type=self.bins_type,
                                                                   bins_type_scale=self.bins_type_scale, concat_masks=self.concat_masks)
            print("[VARIATION PerClassScale]\n")

        #############
        # Instances #
        #############
        elif self.update_block == 3:
            self.update = PerInstanceScale(hidden_dim=self.hidden_dim, context_dim=self.context_dim, bin_num=self.bin_num, \
                                                          loss_type=self.loss_type, num_semantic_classes=self.num_semantic_classes, \
                                                          num_instances=self.num_instances, padding_instances=self.padding_instances, \
                                                          roi_align=roi_align, roi_align_size=roi_align_size, bins_scale=bins_scale, \
                                                          virtual_depth_variation=self.virtual_depth_variation, upsample_type=self.upsample_type, \
                                                          bins_type=self.bins_type, bins_type_scale=self.bins_type_scale, unc_head=self.unc_head)
            print("[VARIATION PerInstanceScale]\n")

        else:
            print("No implementation is available for the given value of update_block. Exiting..")
            exit()

        # Set bacbone layers #
        window_size = int(version[-2:])  # tiny07
        backbone_cfg = dict(embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=window_size, ape=False,
                            drop_path_rate=0.3, patch_norm=True, use_checkpoint=False, frozen_stages=-1)

        # Encoder #
        self.backbone = SwinTransformer(**backbone_cfg)

        # Decoder #
        norm_cfg = dict(type='BN', requires_grad=True)
        embed_dim = 512
        decoder_cfg = dict(in_channels=in_channels, in_index=[0, 1, 2, 3], pool_scales=(1, 2, 3, 6), channels=embed_dim,
                           dropout_ratio=0.0, num_classes=32, norm_cfg=norm_cfg, align_corners=False)

        v_dim = decoder_cfg['num_classes'] * 4
        win = 7
        crf_dims = [128, 256, 512, 1024]
        v_dims = [64, 128, 256, embed_dim]
        self.crf3 = NewCRF(input_dim=in_channels[3], embed_dim=crf_dims[3], window_size=win, v_dim=v_dims[3],
                           num_heads=32)
        self.crf2 = NewCRF(input_dim=in_channels[2], embed_dim=crf_dims[2], window_size=win, v_dim=v_dims[2],
                           num_heads=16)
        self.crf1 = NewCRF(input_dim=in_channels[1], embed_dim=crf_dims[1], window_size=win, v_dim=v_dims[1],
                           num_heads=8)

        # PSP: Pyramid Pooling Module + Convs #
        self.psp_module = PSP(**decoder_cfg)

        # Project decoder features to hidden_dim of update_block #
        self.project = ProjectionV2(v_dims[0], self.hidden_dim)

        # Uncertainty heads #
        if self.unc_head:
            if self.instances_active:
                num_predictions_unc = self.num_instances
            elif self.segmentation_active:
                num_predictions_unc = self.num_semantic_classes
            else:  # Global scale
                num_predictions_unc = 1
            
            # For instances, use RoI masking #
            if not self.instances_active:
                self.unc_head_s = UncertaintyScaleHead(input_dim=crf_dims[0], num_predictions=num_predictions_unc)
            self.unc_head_c = UncertaintyHead(input_dim=crf_dims[0])

        # Initialize layers #
        self.init_weights(pretrained=pretrained)

        # Freeze layers #
        if self.freeze_backbone or self.unc_head:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.psp_module.parameters():
                param.requires_grad = False
            for param in self.crf3.parameters():
                param.requires_grad = False
            for param in self.crf2.parameters():
                param.requires_grad = False

            if self.train_decoder == 0 or self.unc_head:
                for param in self.crf1.parameters():
                    param.requires_grad = False
            else:  # Do not freeze the last decoder layer
                print("[Train last layer of decoder (crf1)]")

            # Freeze complete network from uncertainties #
            if self.unc_head:
                for param in self.project.parameters():
                    param.requires_grad = False

                if not self.instances_active:
                    for param in self.update.parameters():
                        param.requires_grad = False
                else:
                    for name, param in self.update.named_parameters():
                        if 'unc' not in name: 
                            param.requires_grad = False

    def network_info_print(self):
        if self.loss_type == 0:
            print("Using silog loss")
        else:
            print("Using l1 loss")

        if self.update_block == 0:
            if self.loss_type != 0:
                raise ValueError("IEBINS can only be used with silog loss")
            if self.bin_num != 16:
                raise ValueError("IEBINS can only be used with 16 bins")

        if self.upsample_type == 0:
            print("Uncertainty upsampling type: vanilla")
        else:
            print("Uncertainty upsampling type: Rosinol et. al. 2023")

        if self.update_block != 0:
            print("Number of bins (canonical): ", self.bin_num)
            print("Number of bins (scale): ", self.bins_scale)

    def set_to_eval_unc(self):
        """
            Set some layers to eval mode while training the uncertainty heads
        """
        self.backbone.eval()
        self.psp_module.eval()
        self.crf3.eval()
        self.crf2.eval()
        self.crf1.eval()
        self.project.eval()
        if not self.instances_active:
            self.update.eval()
        else:
            for name, module in self.update.named_modules():
                if not "unc" in name:
                    module.eval()

    def init_weights(self, pretrained=None):
        """
        Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
            Defaults to None.
        """
        print(f'== Load encoder backbone from: {pretrained}')
        self.backbone.init_weights(pretrained=pretrained)
        self.psp_module.init_weights()

    def forward(self, imgs, masks=None, instances=None, boxes=None, labels=None):

        if self.segmentation_active and masks is None:
            print("Model mode == Segmentation. Please provide segmentation masks")
            exit()

        if self.instances_active and (not self.segmentation_active or instances is None or boxes is None
                                      or labels is None):
            print("Model mode == Instances. Please provide instance masks")
            exit()

        # Pass to encoder #
        feats = self.backbone(imgs)
        psp_out = self.psp_module(feats)

        # Decoder: CRF concat with bacbone features #
        e3 = self.crf3(feats[3], psp_out)
        e3 = nn.PixelShuffle(2)(e3)
        e2 = self.crf2(feats[2], e3)
        e2 = nn.PixelShuffle(2)(e2)
        e1 = self.crf1(feats[1], e2)
        e1 = nn.PixelShuffle(2)(e1)
        e0 = self.project(e1)  # To hidden_dim

        b, c, h, w = e1.shape
        # Uncertainty predictions using additional heads
        if self.unc_head:
            if not self.instances_active:
                unc_s = self.unc_head_s(e0)
            unc_c = self.unc_head_c(e0)
        
        depth = torch.zeros([b, 1, h, w]).to(e1.device)

        # Ealry feature map - backbone #
        context = feats[0]

        # From crf #
        input_feature_map = torch.tanh(e0)

        #################################################
        # Pass to update_block to predict the depth map #
        #################################################
        result = {}  # result dict

        # Per-Instance or Per-Class #
        if self.segmentation_active or self.instances_active:

            masks = upsample(masks, scale_factor=1 / 4)  # Downsample masks to feature map dim
            if self.instances_active:
                instances = upsample(instances, scale_factor=1 / 4)
                result = self.update(depth, context, input_feature_map, self.bin_num, self.min_depth, self.max_depth, \
                                     masks, instances, boxes, labels)
            else:
                result = self.update(depth, context, input_feature_map, self.bin_num, self.min_depth, self.max_depth,
                                     masks)
        else:  # Global or IEBINS
            result = self.update(depth, context, input_feature_map, self.bin_num, self.min_depth, \
                                 self.max_depth, max_tree_depth=self.max_tree_depth)

        # Uncertainty from extra heads #
        if self.unc_head:
            if not self.instances_active:
                result["unc_s"] = [unc_s]
            result["unc_c"] = [upsample(unc_c, scale_factor=4, upsample_type=self.upsample_type, uncertainty=True)]

        return result


"""
Uncertainty heads 
"""

"""
In: typically 128x120x160
Out: 1xnum_scales
"""


class UncertaintyScaleHead(nn.Module):
    def __init__(self, input_dim=128, num_predictions=1):
        super(UncertaintyScaleHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(64)
        self.fc1 = nn.Linear(64 * 64, num_predictions)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x)).clamp(min=1e-4)
        return x


"""
Out: hxw
"""


class UncertaintyHead(nn.Module):
    def __init__(self, input_dim=128):
        super(UncertaintyHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x)).clamp(min=1e-4)
        return x
