import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer import SwinTransformer
from .newcrf_layers import NewCRF
from .uper_crf_head import PSP
from .depth_update  import *
from .utils import *
########################################################################################################################


class NewCRFDepth(nn.Module):
    """
    Depth network based on neural window FC-CRFs architecture.
    """
    def __init__(self, version=None, inv_depth=False, pretrained=None, 
                    frozen_stages=-1, min_depth=0.1, max_depth=100.0, max_tree_depth=6, bin_num=16, bin_min=0, bin_max=80, update_block=0, loss_type=0, train_decoder=0, **kwargs):
        super().__init__()

        self.inv_depth = inv_depth
        self.with_auxiliary_head = False
        self.with_neck = False
        self.freeze_backbone = True

        self.max_tree_depth = max_tree_depth
        self.bin_num = bin_num
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.update_block = update_block   # 0 iebins, 1 canonical, 2 canonical with probabilities concat in GRU, 3 predict uncertainty map
        self.train_decoder = train_decoder # 0 not train, 1 -> last layer

        self.loss_type = loss_type # 0 for silog 1 for l1


        norm_cfg = dict(type='BN', requires_grad=True)
        window_size = int(version[-2:])

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
       
        # Set update block #
        if self.update_block == 0:
            self.update = BasicUpdateBlockDepth(hidden_dim=128, context_dim=embed_dim, bin_num=16)
        elif self.update_block == 1:
            self.update = BasicUpdateBlockCDepth(hidden_dim=128, context_dim=embed_dim, bin_num=self.bin_num, loss_type=self.loss_type)
        elif self.update_block == 2:
            self.update = BasicUpdateBlockCPDepth(hidden_dim=128, context_dim=embed_dim, bin_num=self.bin_num, loss_type=self.loss_type)
        elif self.update_block == 3:
            self.update = BasicUpdateBlockCUDepth(hidden_dim=128, context_dim=embed_dim, bin_num=self.bin_num, loss_type=self.loss_type)

        backbone_cfg = dict(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=frozen_stages
        )

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

        self.backbone = SwinTransformer(**backbone_cfg)
        v_dim = decoder_cfg['num_classes']*4
        win = 7
        crf_dims = [128, 256, 512, 1024]
        v_dims = [64, 128, 256, embed_dim]
        self.crf3 = NewCRF(input_dim=in_channels[3], embed_dim=crf_dims[3], window_size=win, v_dim=v_dims[3], num_heads=32)
        self.crf2 = NewCRF(input_dim=in_channels[2], embed_dim=crf_dims[2], window_size=win, v_dim=v_dims[2], num_heads=16)
        self.crf1 = NewCRF(input_dim=in_channels[1], embed_dim=crf_dims[1], window_size=win, v_dim=v_dims[1], num_heads=8)

        self.decoder = PSP(**decoder_cfg)

        self.up_mode = 'bilinear'
        if self.up_mode == 'mask':
            self.mask_head = nn.Sequential(
                nn.Conv2d(v_dims[0], 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 16*9, 1, padding=0))

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.hidden_dim = 128
        self.project = Projection(v_dims[0], self.hidden_dim)

        self.init_weights(pretrained=pretrained)

	    # Freeze some weights #
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.crf3.parameters():
                param.requires_grad = False
            for param in self.crf2.parameters():
                param.requires_grad = False
            #for param in self.crf1.parameters():
            #    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        print(f'== Load encoder backbone from: {pretrained}')
        self.backbone.init_weights(pretrained=pretrained)
        self.decoder.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def upsample_mask(self, disp, mask):
        """ Upsample disp [H/4, W/4, 1] -> [H, W, 1] using convex combination """
        N, C, H, W = disp.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(disp, kernel_size=3, padding=1)
        up_disp = up_disp.view(N, C, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, C, 4*H, 4*W)

    def forward(self, imgs, epoch=1, step=100):

        feats = self.backbone(imgs)
        ppm_out = self.decoder(feats)

        e3 = self.crf3(feats[3], ppm_out)
        e3 = nn.PixelShuffle(2)(e3)
        e2 = self.crf2(feats[2], e3)
        e2 = nn.PixelShuffle(2)(e2)
        e1 = self.crf1(feats[1], e2)
        e1 = nn.PixelShuffle(2)(e1)

        max_tree_depth = self.max_tree_depth

        if self.up_mode == 'mask':
            mask = self.mask_head(e1)

        b, c, h, w = e1.shape
        device = e1.device
 
        depth = torch.zeros([b, 1, h, w]).to(device)
        context = feats[0]
        gru_hidden = torch.tanh(self.project(e1))
        # Hidden initialized with decoder

        result = self.update(depth, context, gru_hidden, max_tree_depth, self.bin_num, self.bin_min, self.bin_max)

        if self.up_mode == 'mask':
            for i in range(max_tree_depth):
                result["pred_depths_r_list"][i] = self.upsample_mask(result["pred_depths_r_list"][i], mask)  
                result["pred_depths_c_list"][i] = self.upsample_mask(result["pred_depths_c_list"][i], mask.detach())
                result["pred_depths_rc_list"][i] = self.upsample_mask(result["pred_depths_rc_list"][i], mask.detach())
                result["uncertainty_maps_list"][i] = self.upsample_mask(result["uncertainty_maps_list"][i], mask.detach())
            if self.update_block == 3:        
                for i in range(max_tree_depth):
                    result["pred_depths_u_list"][i] = self.upsample_mask(result["pred_depths_u_list"][i], mask.detach())
                   
        else:
            for i in range(max_tree_depth):
                result["pred_depths_r_list"][i] = upsample(result["pred_depths_r_list"][i], scale_factor=4)
                result["pred_depths_c_list"][i] = upsample(result["pred_depths_c_list"][i], scale_factor=4) 
                result["pred_depths_rc_list"][i] = upsample(result["pred_depths_rc_list"][i], scale_factor=4) 
                result["uncertainty_maps_list"][i] = upsample(result["uncertainty_maps_list"][i], scale_factor=4) 
            if self.update_block == 3:        
                for i in range(max_tree_depth):
                    result["pred_depths_u_list"][i] = upsample(result["pred_depths_u_list"][i], scale_factor=4)

        return result 
