import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer import SwinTransformer
from .newcrf_layers import NewCRF
from .uper_crf_head import PSP
from .depth_update  import *
from .utils import *


class NewCRFDepth(nn.Module):
    """
    Depth network based on neural window FC-CRFs architecture
    """
    def __init__(self, version=None, pretrained=None, 
                    frozen_stages=-1, min_depth=0.1, max_depth=100.0, max_tree_depth=6, 
                    bin_num=16, update_block=0, loss_type=0, train_decoder=0, 
                    predict_unc=False, predict_unc_d3vo=False, **kwargs):
        super().__init__()

        self.with_auxiliary_head = False
        self.with_neck = False
        
        # 1 train last layer of decoder 
        self.freeze_backbone = True
        self.train_decoder = train_decoder 
        
        # GRU iter
        self.max_tree_depth = max_tree_depth 
        self.bin_num = bin_num
        self.min_depth = min_depth
        self.max_depth = max_depth
      
        # Uncertainty
        self.predict_unc = predict_unc
        self.predict_unc_d3vo = predict_unc_d3vo

        # 0 silog loss relu 
        self.update_block = update_block  
        if loss_type == 0:
            self.loss_type = 0 
        else:
            self.loss_type = 1
        
        norm_cfg = dict(type='BN', requires_grad=True)
        window_size = int(version[-2:])

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
       
        # Set update block #
        if self.update_block == 0: # IEBins
            self.update = BasicUpdateBlockDepth(hidden_dim=128, context_dim=embed_dim, bin_num=16)
        elif self.update_block == 1: # Canonical - one scale per pixel
            self.update = BasicUpdateBlockCDepth(hidden_dim=128, context_dim=embed_dim, bin_num=self.bin_num, loss_type=self.loss_type)
        elif self.update_block == 2: # with uncertainty prediction (from GRU) 
            self.update = BasicUpdateBlockCUDepth(hidden_dim=128, context_dim=embed_dim, bin_num=self.bin_num, loss_type=self.loss_type)
        elif self.update_block == 3: # Canonical - one scale with uncertainty (from decoder) concatenation
            self.update = BasicUpdateBlockCUConcDepth(hidden_dim=128, context_dim=embed_dim, bin_num=self.bin_num, loss_type=self.loss_type)
        elif self.update_block == 4: # Canonical. one scale per image 
            self.update = BasicUpdateBlockCSDepth(hidden_dim=128, context_dim=embed_dim, bin_num=self.bin_num, loss_type=self.loss_type)
        else:
            pass

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
       
        # Decoder layers # 
        v_dim = decoder_cfg['num_classes']*4
        win = 7
        crf_dims = [128, 256, 512, 1024]
        v_dims = [64, 128, 256, embed_dim]
        self.crf3 = NewCRF(input_dim=in_channels[3], embed_dim=crf_dims[3], window_size=win, v_dim=v_dims[3], num_heads=32)
        self.crf2 = NewCRF(input_dim=in_channels[2], embed_dim=crf_dims[2], window_size=win, v_dim=v_dims[2], num_heads=16)
        self.crf1 = NewCRF(input_dim=in_channels[1], embed_dim=crf_dims[1], window_size=win, v_dim=v_dims[1], num_heads=8)
        #self.crf0 = NewCRF(input_dim=in_channels[0], embed_dim=crf_dims[0], window_size=win, v_dim=v_dims[0], num_heads=4) # Used in NDDepth 

        # Last layer - "downsampling" encoder # 
        self.psp_module = PSP(**decoder_cfg)

        # Depth upsampling #
        self.up_mode = 'bilinear'
        if self.up_mode == 'mask':
            self.mask_head = nn.Sequential(
                nn.Conv2d(v_dims[0], 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 16*9, 1, padding=0))

        # GRU #
        self.hidden_dim = 128
        self.project = Projection(v_dims[0], self.hidden_dim) # Project features to GRU input  

        # Predict uncertainty from decoder features #
        if self.predict_unc: 
            self.uncer_head = UncerHead(input_dim=crf_dims[0])
        if self.predict_unc_d3vo:
            self.uncer_d3vo_head = D3VOUncerHead(input_dim=crf_dims[0])
 
        # Initialize layers #
        self.init_weights(pretrained=pretrained)

	    # Freeze some weights #
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.psp_module.parameters():
                param.requires_grad = False
            for param in self.crf3.parameters():
                param.requires_grad = False
            for param in self.crf2.parameters():
                param.requires_grad = False
            if self.train_decoder == 0:
                for param in self.crf1.parameters():
                    param.requires_grad = False
            else:
                print("Training last decoder layer")

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        print(f'== Load encoder backbone from: {pretrained}')
        self.backbone.init_weights(pretrained=pretrained)
        self.psp_module.init_weights()
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
        psp_out = self.psp_module(feats)

        # crf concat with encoder features
        e3 = self.crf3(feats[3], psp_out)
        e3 = nn.PixelShuffle(2)(e3)
        e2 = self.crf2(feats[2], e3)
        e2 = nn.PixelShuffle(2)(e2)
        e1 = self.crf1(feats[1], e2)
        e1 = nn.PixelShuffle(2)(e1) 
        e0 = self.project(e1)
        #e0 = self.crf0(feats[0], e1) # NDDepth - remove tanh after? we also need project after btw 

        max_tree_depth = self.max_tree_depth

        if self.up_mode == 'mask':
            mask = self.mask_head(e1)

        # Uncertainty prediction from the decoder #
        if self.predict_unc:
            unc = self.uncer_head(e0)
        elif self.predict_unc_d3vo:
            unc_d3vo = self.uncer_d3vo_head(e0)

        b, c, h, w = e1.shape
        device = e1.device
 
        depth = torch.zeros([b, 1, h, w]).to(device)
        context = feats[0]
        gru_hidden = torch.tanh(e0)

        # Predict depth with GRU. context: early feature map and hidden: late feature map #
        if self.predict_unc == False and self.update_block != 3:
            result = self.update(depth, context, gru_hidden, max_tree_depth, self.bin_num, self.min_depth, self.max_depth)
        else:
            result = self.update(depth, unc, context, gru_hidden, max_tree_depth, self.bin_num, self.min_depth, self.max_depth)

        if self.up_mode == 'mask':
            for i in range(max_tree_depth):
                result["pred_depths_r_list"][i] = self.upsample_mask(result["pred_depths_r_list"][i], mask)  
                result["pred_depths_c_list"][i] = self.upsample_mask(result["pred_depths_c_list"][i], mask.detach())
                result["pred_depths_rc_list"][i] = self.upsample_mask(result["pred_depths_rc_list"][i], mask.detach())
                result["uncertainty_maps_list"][i] = self.upsample_mask(result["uncertainty_maps_list"][i], mask.detach())

            if self.update_block == 2: # Predict uncertainty from GRU       
                for i in range(max_tree_depth):
                    result["pred_depths_u_list"][i] = self.upsample_mask(result["pred_depths_u_list"][i], mask.detach())

            if self.predict_unc:
                unc = self.upsample_mask(unc, mask.detach())
                result["unc"] = unc
            elif self.predict_unc_d3vo:
                unc_d3vo = self.upsample_mask(unc_d3vo, mask.detach())
                result["unc_d3vo"] = unc_d3vo
        else:
            for i in range(max_tree_depth):
                result["pred_depths_r_list"][i] = upsample(result["pred_depths_r_list"][i], scale_factor=4)
                result["pred_depths_c_list"][i] = upsample(result["pred_depths_c_list"][i], scale_factor=4) 
                result["pred_depths_rc_list"][i] = upsample(result["pred_depths_rc_list"][i], scale_factor=4) 
                result["uncertainty_maps_list"][i] = upsample(result["uncertainty_maps_list"][i], scale_factor=4) 

            if self.update_block == 2:  # Predict uncertainty from GRU      
                for i in range(max_tree_depth):
                    result["pred_depths_u_list"][i] = upsample(result["pred_depths_u_list"][i], scale_factor=4)
            if self.predict_unc:
                unc = upsample(unc, scale_factor=4)
                result["unc"] = unc
            elif self.predict_unc_d3vo:
                unc_d3vo = upsample(unc_d3vo, scale_factor=4)
                result["unc_d3vo"] = unc_d3vo

        return result

class UncerHead(nn.Module):
    """
    NDDepth uncertainty head [0,1]
    """
    def __init__(self, input_dim=100):
        super(UncerHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.conv1(x))
        return x 

class D3VOUncerHead(nn.Module):
    """
    D3VO uncertainty head [0, inf]
    """
    def __init__(self, input_dim=100):
        super(D3VOUncerHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = (self.relu1(self.conv1(x))).clamp(min=1e-3)
        return x 
