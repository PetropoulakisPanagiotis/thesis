import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer import SwinTransformer
from .newcrf_layers import NewCRF
from .uper_crf_head import PSP
from .depth_update_clean_scale  import *
from .utils_clean import *


class NewCRFDepth(nn.Module):
    """
    Depth network based on neural window FC-CRFs architecture
    """
    def __init__(self, version=None, pretrained=None, min_depth=0.1, max_depth=100.0, max_tree_depth=1, 
                    bin_num=16, update_block=0, loss_type=0, train_decoder=0, predict_unc=False, \
                    num_semantic_classes=14, num_instances=63, var=0, padding_instances=0, \
                    segmentation_active= False, instances_active=False, roi_align=False, roi_align_size=32, \
                    bins_scale=50, d3vo=False, **kwargs):
        super().__init__()
        self.freeze_backbone = True
        self.train_decoder = train_decoder # train the last layer of decoder 
        
        # Bins
        self.max_tree_depth = max_tree_depth 
        self.bin_num = bin_num
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        # Instances 
        self.segmentation_active = segmentation_active
        self.instances_active = instances_active
        
        self.num_semantic_classes = num_semantic_classes
        self.num_instances = num_instances
        self.var = var
        self.padding_instances = padding_instances

        # Uncertainty from decoder head
        self.predict_unc = predict_unc

        # Uncertainty d3vo
        self.d3vo = d3vo

        # 0 silog (use relu) or l1 
        self.update_block = update_block  
        if loss_type == 0:
            self.loss_type = 0 
        else:
            self.loss_type = 1
        
        # Backbone dims
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

        # Set update block for depth prediction head
        if self.update_block == 0: 
            self.update = IEBINS(hidden_dim=self.hidden_dim, context_dim=self.context_dim, bin_num=bin_num)
            print("[VARIATION IEBINS]\n")   

        elif self.update_block == 4: 
            self.update = Regression(hidden_dim=self.hidden_dim, context_dim=self.context_dim, \
                                                loss_type=self.loss_type)
            print("[VARIATION Regression]\n")
   
        elif self.update_block == 8: 
            self.update = RegressionSingleScale(hidden_dim=self.hidden_dim, context_dim=self.context_dim, \
                                                loss_type=self.loss_type)
            print("[VARIATION RegressionSingleScale]\n")

        elif self.update_block == 3:
            self.update = Uniform(hidden_dim=self.hidden_dim, context_dim=self.context_dim, bin_num=self.bin_num)
            print("[VARIATION Uniform]\n")
        
        elif self.update_block == 18:
            self.update = UniformSingleScale(hidden_dim=self.hidden_dim, context_dim=self.context_dim, bin_num=self.bin_num, loss_type=self.loss_type, bins_scale=bins_scale)
            print("[VARIATION UniformSingleScale]\n")
        
        ################
        # Segmentation #
        ################        
        elif self.update_block == 1:  
            self.update = UniformSegmentationModuleListConcatMasks(hidden_dim=self.hidden_dim, context_dim=self.context_dim, bin_num=self.bin_num, \
                                                                   loss_type=self.loss_type, num_semantic_classes=self.num_semantic_classes, bins_scale=bins_scale)                
            print("[VARIATION UniformSegmentationModuleListConcatMasks]\n")

        elif self.update_block == 12: 
            self.update = RegressionSegmentationNoMasking(hidden_dim=self.hidden_dim, context_dim=self.context_dim, \
                                                          loss_type=self.loss_type, num_semantic_classes=self.num_semantic_classes)        
            print("[VARIATION RegressionSegmentationNoMasking]\n")
        
        elif self.update_block == 13: 
            self.update = RegressionSegmentationNoMaskingConcatMasks(hidden_dim=self.hidden_dim, context_dim=self.context_dim, \
                                                                     loss_type=self.loss_type, num_semantic_classes=self.num_semantic_classes)
            print("[VARIATION RegressionSegmentationNoMaskingConcatMasks]\n")
        
        elif self.update_block == 15: 
            self.update = RegressionSegmentationModuleListConcatMasks(hidden_dim=self.hidden_dim, context_dim=self.context_dim, \
                                                                           loss_type=self.loss_type, \
                                                                           num_semantic_classes=self.num_semantic_classes)
            print("[VARIATION RegressionSegmentationModuleListConcatMasks]\n")
        
        #############
        # Instances #
        #############
        elif self.update_block == 2: 
            self.update = UniformInstancesSharedCanonical(hidden_dim=self.hidden_dim, context_dim=self.context_dim, bin_num=self.bin_num, \
                                                          loss_type=self.loss_type, num_semantic_classes=self.num_semantic_classes, \
                                                          num_instances=self.num_instances, var=var, padding_instances=self.padding_instances, \
                                                          roi_align=roi_align, roi_align_size=roi_align_size, bins_scale=bins_scale)        
            print("[VARIATION UniformInstancesSharedCanonical]\n")
        
        elif self.update_block == 20: 
            self.update = RegressionInstances(hidden_dim=self.hidden_dim, context_dim=self.context_dim, \
                                              loss_type=self.loss_type, num_semantic_classes=self.num_semantic_classes, \
                                              num_instances=self.num_instances, var=var, padding_instances=self.padding_instances)         
            print("[VARIATION RegressionInstances]\n")
        
        elif self.update_block == 22: 
            self.update = RegressionInstancesSharedCanonical(hidden_dim=self.hidden_dim, context_dim=self.context_dim, \
                                                             loss_type=self.loss_type, num_semantic_classes=self.num_semantic_classes, \
                                                             num_instances=self.num_instances, var=var, padding_instances=self.padding_instances)        
            print("[VARIATION RegressionInstancesSharedCanonical]\n")
        
        elif self.update_block == 23: 
            self.update = RegressionInstancesSharedCanonicalClass(hidden_dim=self.hidden_dim, context_dim=self.context_dim, \
                                                                  loss_type=self.loss_type, num_semantic_classes=self.num_semantic_classes, \
                                                                  num_instances=self.num_instances, var=var, padding_instances=self.padding_instances)        
            print("[VARIATION RegressionInstancesSharedCanonicalClass]\n")
        

        elif self.update_block == 25: 
            self.update = RegressionInstancesSharedCanonicalModuleScale(hidden_dim=self.hidden_dim, context_dim=self.context_dim, \
                                                                        loss_type=self.loss_type, num_semantic_classes=self.num_semantic_classes, \
                                                                        num_instances=self.num_instances, var=var, padding_instances=self.padding_instances)        
            print("[VARIATION RegressionInstancesSharedCanonicalModuleScale]\n")
        
        elif self.update_block == 26: 
            self.update = RegressionInstancesModule(hidden_dim=self.hidden_dim, context_dim=self.context_dim, \
                                                    loss_type=self.loss_type, num_semantic_classes=self.num_semantic_classes, \
                                                    num_instances=self.num_instances, var=var, padding_instances=self.padding_instances)        
            print("[VARIATION RegressionInstancesModule]\n")
        
        else:
            print("No implementation is available for the given number of update_block. Exit")
            exit()

        window_size = int(version[-2:]) # tiny07
        backbone_cfg = dict(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=-1
        )

        self.backbone = SwinTransformer(**backbone_cfg)
        
        norm_cfg = dict(type='BN', requires_grad=True)
        embed_dim = 512
        decoder_cfg = dict(
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=embed_dim,
            dropout_ratio=0.0,
            num_classes=32,
            norm_cfg=norm_cfg,
            align_corners=False
        )

       
        # Decoder layers
        v_dim = decoder_cfg['num_classes']*4
        win = 7
        crf_dims = [128, 256, 512, 1024]
        v_dims = [64, 128, 256, embed_dim]
        self.crf3 = NewCRF(input_dim=in_channels[3], embed_dim=crf_dims[3], window_size=win, v_dim=v_dims[3], num_heads=32)
        self.crf2 = NewCRF(input_dim=in_channels[2], embed_dim=crf_dims[2], window_size=win, v_dim=v_dims[2], num_heads=16)
        self.crf1 = NewCRF(input_dim=in_channels[1], embed_dim=crf_dims[1], window_size=win, v_dim=v_dims[1], num_heads=8)

        # PSP: Pyramid Pooling Module + Convs 
        self.psp_module = PSP(**decoder_cfg)

        # Project decoder features to hidden_dim of update_block
        self.project = ProjectionV2(v_dims[0], self.hidden_dim) 

        # Predict uncertainty from the decoder features
        if self.predict_unc: 
            self.uncer_head = UncertaintyHead(input_dim=crf_dims[0])

        # D3VO uncertainty  
        if self.d3vo:
            if self.instances_active:
                num_classes = self.num_instances
            elif self.segmentation_active:
                num_classes = self.num_semantic_classes
            else:
                num_classes = 1

            self.d3vo_head = D3VOHead(input_dim=crf_dims[0], num_classes=num_classes) 

        # Initialize layers
        self.init_weights(pretrained=pretrained)

	    # Freeze some layers
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.psp_module.parameters():
                param.requires_grad = False
            for param in self.crf3.parameters():
                param.requires_grad = False
            for param in self.crf2.parameters():
                param.requires_grad = False
            if self.train_decoder == 0 or self.d3vo:
                for param in self.crf1.parameters():
                    param.requires_grad = False
            else:
                print("[Train last layer of decoder (crf1)]")
        
        if self.d3vo:
            for param in self.project.parameters():
                param.requires_grad = False

            for param in self.update.parameters():
                param.requires_grad = False

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

        if self.segmentation_active and masks == None:
            print("Model mode == Segmentation. Please provide segmentation masks")
            exit()

        if self.instances_active and (instances == None or boxes == None or labels == None):
            print("Model mode == Instances. Please provide instance masks")
            exit()

        feats = self.backbone(imgs)
        psp_out = self.psp_module(feats)

        # CRF concat with bacbone features
        e3 = self.crf3(feats[3], psp_out)
        e3 = nn.PixelShuffle(2)(e3)
        e2 = self.crf2(feats[2], e3)
        e2 = nn.PixelShuffle(2)(e2)
        e1 = self.crf1(feats[1], e2)
        e1 = nn.PixelShuffle(2)(e1) 
        e0 = self.project(e1)      # To hidden_dim  

        # Uncertainty prediction from the decoder
        if self.predict_unc:
            unc_decoder = self.uncer_head(e0)

        if self.d3vo:
            unc_d3vo = self.d3vo_head(e0)

        b, c, h, w = e1.shape

        depth = torch.zeros([b, 1, h, w]).to(e1.device)

        # Ealry feature map - backbone
        context = feats[0] 
        
        # From crf
        input_feature_map = torch.tanh(e0) 


        #################################################
        # Pass to update_block to predict the depth map #
        #################################################

        # Segmentation/Instances #
        if self.segmentation_active or self.instances_active:
            # Downsample masks to feature map dim
            masks = upsample(masks, scale_factor=1/4)
            
            if self.instances_active:
                instances = upsample(instances, scale_factor=1/4)
                result = self.update(depth, context, input_feature_map, self.bin_num, self.min_depth, self.max_depth, \
                                     masks, instances, boxes, labels)
            else:
                result = self.update(depth, context, input_feature_map, self.bin_num, self.min_depth, self.max_depth, masks)
        # Vanilla #
        else:
            if self.update_block == 0:
                result = self.update(depth, context, input_feature_map, self.bin_num, self.min_depth, \
                                     self.max_depth, max_tree_depth=self.max_tree_depth)
            else:
                result = self.update(depth, context, input_feature_map, self.bin_num, self.min_depth, self.max_depth)


        # Process result and upsample # 
        for i in range(self.max_tree_depth):

            if not self.instances_active:
                result["pred_depths_r_list"][i] = upsample(result["pred_depths_r_list"][i], scale_factor=4)
                
                # Canonical prediction #
                if self.update_block == 1 or self.update_block == 8 or self.update_block == 18 or self.update_block == 12 or \
                   self.update_block == 13 or self.update_block == 15:
                    result["pred_depths_rc_list"][i] = upsample(result["pred_depths_rc_list"][i], scale_factor=4) 
            else:
                result["pred_depths_instances_r_list"][i] = upsample(result["pred_depths_instances_r_list"][i], scale_factor=4)  
                result["pred_depths_instances_rc_list"][i] = upsample(result["pred_depths_instances_rc_list"][i], scale_factor=4)

               
            # Uncertainty via bins #
            if self.update_block == 0 or self.update_block == 3 or self.update_block == 18 or\
               self.update_block == 1 or self.update_block == 2:
                result["pred_depths_c_list"][i] = upsample(result["pred_depths_c_list"][i], scale_factor=4) 
                result["uncertainty_maps_list"][i] = upsample(result["uncertainty_maps_list"][i], scale_factor=4) 

            # Uncertainty from the decoder block #
            if self.predict_unc:
                unc_decoder = upsample(unc_decoder, scale_factor=4)
                result["unc_decoder"] = unc_decoder

            if self.d3vo:
                result["unc_d3vo"] = unc_d3vo

        return result


"""
UncertaintyHead [0, 1]
"""
class UncertaintyHead(nn.Module):
    def __init__(self, input_dim=128):
        super(UncertaintyHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.conv1(x))
        return x 


"""
D3VO unc
"""
class D3VOHead(nn.Module):
    def __init__(self, input_dim=128, num_classes=1):
        super(D3VOHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)

        self.pool = nn.AdaptiveAvgPool2d(88)
        self.fc1 = nn.Linear(88 * 88, num_classes)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        print(x.shape)
        return x 
