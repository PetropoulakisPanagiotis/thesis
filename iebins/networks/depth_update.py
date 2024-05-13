import torch
import torch.nn as nn
import torch.nn.functional as F

padding_global = 0

from .depth_update_heads import *
from .utils import *
from torchvision.ops import RoIAlign


"""
IEBins implementation 
"""
class IEBINS(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16):
        super(IEBINS, self).__init__()

        self.encoder_project = ProjectionInputDepth(hidden_dim=hidden_dim, out_chs=hidden_dim * 2, bin_num=bin_num)

        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=self.encoder_project.out_chs + context_dim)

        self.p_head = PHead(hidden_dim, hidden_dim, bin_num)

    def forward(self, depth, context, gru_hidden, bin_num, min_depth, max_depth, max_tree_depth=6):

        pred_depths_r_list = [] # Metric
        pred_depths_c_list = [] # Labels
        uncertainty_maps_list = []

        # Create a feature map of size depth with the bin canditates values
        bin_edges, current_depths = get_iebins(depth, min_depth, max_depth, bin_num)

        for i in range(max_tree_depth):
            input_features = self.encoder_project(current_depths.detach())
            input_c = torch.cat([input_features, context], dim=1)

            gru_hidden = self.gru(gru_hidden, input_c)
            pred_prob = self.p_head(gru_hidden)

            # Metric
            depth_r = (pred_prob * current_depths.detach()).sum(1, keepdim=True)
            pred_depths_r_list.append(depth_r)

            uncertainty_map = torch.sqrt((pred_prob * ((current_depths.detach() - depth_r.repeat(1, bin_num, 1, 1))**2)).sum(1, keepdim=True))
            uncertainty_maps_list.append(uncertainty_map)

            pred_label = get_label(torch.squeeze(depth_r, 1), bin_edges, bin_num).unsqueeze(1)
            depth_c = torch.gather(current_depths.detach(), 1, pred_label.detach())
            pred_depths_c_list.append(depth_c)
            
            label_target_bin_left = pred_label
            target_bin_left = torch.gather(bin_edges, 1, label_target_bin_left)
            label_target_bin_right = (pred_label.float() + 1).long()
            target_bin_right = torch.gather(bin_edges, 1, label_target_bin_right)

            bin_edges, current_depths = update_bins(bin_edges, target_bin_left, target_bin_right, depth_r.detach(), pred_label.detach(), \
                                                    bin_num, min_depth, max_depth, uncertainty_map)
        

        result = {}        
        result["pred_depths_r_list"] = pred_depths_r_list
        result["pred_depths_c_list"] = pred_depths_c_list
        result["uncertainty_maps_list"] = uncertainty_maps_list

        return result


"""
Regression prediction 
"""
class Regression(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, loss_type=0):
        super(Regression, self).__init__()

        self.r_head = RegressionHead(hidden_dim, hidden_dim) # Canonical regression [0, 1]

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth):
        pred_depths_r_list = []    # Metric 
         
        depth_r = self.r_head(input_feature_map)        
        
        # Metric
        if self.loss_type == 0:
            depth_r = self.relu(depth_r).clamp(min=1e-3)

        pred_depths_r_list.append(depth_r)

        result = {}
        result["pred_depths_r_list"] = pred_depths_r_list

        return result


"""
Regression prediction with single scale per image 
"""
class RegressionSingleScale(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, loss_type=0):
        super(RegressionSingleScale, self).__init__()

        self.sigmoid_head = SigmoidHead(hidden_dim, hidden_dim) # Canonical regression [0, 1]
        self.s_head = SSHead(hidden_dim, 2)            # Global scale and shift 

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth):
        pred_depths_r_list = []    # Metric 
        pred_depths_rc_list = []   # Canonical

        pred_scale_list = []
        pred_shift_list = []
        
         
        pred_rc = self.sigmoid_head(input_feature_map)        
        pred_scale_shift = self.s_head(input_feature_map)

        pred_scale_list.append(pred_scale_shift[:, 0:1])
        pred_shift_list.append(pred_scale_shift[:, 1:2])
        
        # Canonical
        depth_rc = pred_rc
        pred_depths_rc_list.append(depth_rc)

        # Metric
        if self.loss_type == 0:
            depth_r = (self.relu(depth_rc * pred_scale_shift[:, 0:1].unsqueeze(1).unsqueeze(1) + pred_scale_shift[:, 1:2].unsqueeze(1).unsqueeze(1))).clamp(min=1e-3)
        else:
            depth_r = depth_rc * pred_scale_shift[:, 0:1].unsqueeze(1).unsqueeze(1) + pred_scale_shift[:, 1:2].unsqueeze(1).unsqueeze(1)

        pred_depths_r_list.append(depth_r)

        result = {}
        result["pred_depths_r_list"] = pred_depths_r_list
        result["pred_depths_rc_list"] = pred_depths_rc_list
        result["pred_scale_list"] = pred_scale_list
        result["pred_shift_list"] = pred_shift_list

        return result


"""
Uniform bins
"""
class Uniform(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16):
        super(Uniform, self).__init__()

        self.p_head = PHead(hidden_dim, hidden_dim, bin_num + 1) # Include 0

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth):
        pred_depths_r_list = [] # Metric 

        bins_map = get_uniform_bins(depth, min_depth, max_depth, bin_num)

        pred_scale_list = []
        pred_shift_list = []
        uncertainty_maps_list = []
        pred_depths_c_list = [] 
        
        pred_r = self.p_head(input_feature_map)        
       
        depth_r = (pred_r * bins_map.detach()).sum(1, keepdim=True)
        pred_depths_r_list.append(depth_r)

        uncertainty_map = torch.sqrt((pred_r * ((bins_map.detach() - depth_r.repeat(1, bin_num+1, 1, 1))**2)).sum(1, keepdim=True))
        uncertainty_maps_list.append(uncertainty_map)

        pred_label = get_label(torch.squeeze(depth_r, 1), bins_map, bin_num).unsqueeze(1)
        depth_c = torch.gather(bins_map.detach(), 1, pred_label.detach())
        pred_depths_c_list.append(depth_c)

        result = {}
        result["pred_depths_r_list"] = pred_depths_r_list
        result["pred_depths_c_list"] = pred_depths_c_list
        result["uncertainty_maps_list"] = uncertainty_maps_list

        return result


"""
Uniform bins with single scale per image
"""
class UniformSingleScale(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, bins_scale=50, virtual_depth_variation=0):
        super(UniformSingleScale, self).__init__()
        self.loss_type = loss_type
        self.bins_scale = bins_scale
        self.virtual_depth_variation = virtual_depth_variation
        self.p_head = PHead(hidden_dim, hidden_dim, bin_num + 1)  # Propabilities canonical

        # Global scale and shift regression
        if self.virtual_depth_variation == 0:
            self.s_head = SSHead(hidden_dim, 2)                    
        # Global scale regression
        elif self.virtual_depth_variation == 1:
            self.s_head = SSHead(hidden_dim, 1)                    
        # Global scale binning
        else:
            self.s_head = SHead(hidden_dim, num_bins=bins_scale + 1)  
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth):
        pred_depths_r_list = []    # Metric 
        pred_depths_rc_list = []   # Canonical

        bins_map = get_uniform_bins(depth, min_depth, max_depth, bin_num)
      
        bins_map_scale = get_uniform_bins(torch.zeros(depth.shape[0], 1, 1, 1).to(depth.device), 0, 15, self.bins_scale).squeeze(-1).squeeze(-1)
       
        if self.virtual_depth_variation == 2: 
            bins_map_shift = get_uniform_bins(torch.zeros(depth.shape[0], 1, 1, 1).to(depth.device), 0, 2, 50).squeeze(-1).squeeze(-1)
        
        pred_scale_list = []
        pred_shift_list = []
        uncertainty_maps_list = []
        pred_depths_c_list = [] 
        pred_depths_scale_c_list = [] 
        uncertainty_maps_scale_list = []
        
        # Canonical propabilities 
        pred_rc = self.p_head(input_feature_map)       
        
        # Scale and shift 
        pred_scale_shift = self.s_head(input_feature_map)
        if self.virtual_depth_variation == 0:
            pred_scale_list.append(pred_scale_shift[:, 0:1])
            pred_shift_list.append(pred_scale_shift[:, 1:2])
        elif self.virtual_depth_variation == 1:
            pred_scale_list.append(pred_scale_shift)
        else:
            # Scale prob #
            scale = (pred_scale_shift * bins_map_scale.detach()).sum(1, keepdim=True)
            pred_scale_list.append(scale)

            uncertainty_map = torch.sqrt((pred_scale_shift * ((bins_map_scale.detach() - scale.repeat(1, self.bins_scale+1, 1, 1))**2)).sum(1, keepdim=True))
            uncertainty_maps_scale_list.append(uncertainty_map)
        
            # Label
            pred_label = get_label(torch.squeeze(scale, 1), bins_map_scale, self.bins_scale + 1).unsqueeze(1)
            depth_scale_c = torch.gather(bins_map_scale.detach(), 1, pred_label.detach())
            pred_depths_scale_c_list.append(depth_scale_c)
        
        # Canonical
        depth_rc = (pred_rc * bins_map.detach()).sum(1, keepdim=True)
        pred_depths_rc_list.append(depth_rc)

        uncertainty_map = torch.sqrt((pred_rc * ((bins_map.detach() - depth_rc.repeat(1, bin_num+1, 1, 1))**2)).sum(1, keepdim=True))
        uncertainty_maps_list.append(uncertainty_map)
      
        # Label
        pred_label = get_label(torch.squeeze(depth_rc, 1), bins_map, bin_num).unsqueeze(1)
        depth_c = torch.gather(bins_map.detach(), 1, pred_label.detach())
        pred_depths_c_list.append(depth_c)

        # Metric
        if self.loss_type == 0:
            if self.virtual_depth_variation == 0:
                depth_r = (self.relu(depth_rc * pred_scale_list[-1].unsqueeze(1).unsqueeze(1) + pred_shift_list[-1].unsqueeze(1).unsqueeze(1))).clamp(min=1e-3)
            else:
                depth_r = (self.relu(depth_rc * pred_scale_list[-1].unsqueeze(1).unsqueeze(1))).clamp(min=1e-3)
        else:
            if self.virtual_depth_variation == 0:
                depth_r = depth_rc * pred_scale_list[-1].unsqueeze(1).unsqueeze(1) + pred_shift_list[-1].unsqueeze(1).unsqueeze(1)
            else:
                depth_r = depth_rc * pred_scale_list[-1].unsqueeze(1).unsqueeze(1)

        pred_depths_r_list.append(depth_r)

        result = {}
        result["pred_depths_r_list"] = pred_depths_r_list
        result["pred_depths_rc_list"] = pred_depths_rc_list
        result["pred_depths_c_list"] = pred_depths_c_list
        result["pred_depths_scale_c_list"] = pred_depths_scale_c_list
        result["uncertainty_maps_list"] = uncertainty_maps_list
        result["uncertainty_maps_scale_list"] = uncertainty_maps_scale_list
        result["pred_scale_list"] = pred_scale_list
        result["pred_shift_list"] = pred_shift_list

        return result


"""
Segmentation variations
"""

class UniformSegmentationModuleListConcatMasks(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=5, bins_scale=50, virtual_depth_variation=0):
        super(UniformSegmentationModuleListConcatMasks, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.loss_type = loss_type
        self.bins_scale = bins_scale
        self.virtual_depth_variation = virtual_depth_variation
        
        self.relu = nn.ReLU(inplace=True)

        self.p_heads = nn.ModuleList()
        self.s_heads = nn.ModuleList()
        for i in range(num_semantic_classes):       
            self.p_heads.append(PHead(128 + 1, 128, bin_num=bin_num+1))
            
            if self.virtual_depth_variation == 0:
                self.s_heads.append(SSHead(128 + 1, num_out=2))                
            elif self.virtual_depth_variation == 1:
                self.s_heads.append(SSHead(128 + 1, num_out=1))                
            elif self.virtual_depth_variation == 2:
                self.s_heads.append(SHead(128 + 1, num_bins=self.bins_scale+1))                

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, masks):
        pred_depths_r_list = []    # Metric 
        pred_depths_rc_list = []   # Canonical

        pred_scale_list = []
        pred_shift_list = []

        uncertainty_maps_list = []
        uncertainty_maps_scale_list = []
        pred_depths_c_list = [] 
        pred_depths_scale_c_list = [] 

        b, _, h, w = depth.shape
        depth_rc = []
        pred_scale_shift = [] 

        bins_map = get_uniform_bins(depth, min_depth, max_depth, bin_num)
        if self.virtual_depth_variation == 2:
            bins_map_scale = get_uniform_bins(torch.zeros(depth.shape[0], 1, 1, 1).to(depth.device), 0, 15, self.bins_scale).squeeze(-1).squeeze(-1)

        uncertainty_map = torch.zeros(b, self.num_semantic_classes, h, w).to(masks.device)
        depth_c = torch.zeros(b, self.num_semantic_classes, h, w).to(masks.device)
        
        if self.virtual_depth_variation == 2:
            uncertainty_map_scale = torch.zeros(b, self.num_semantic_classes, 1, 1).to(masks.device)
            c_map_scale = torch.zeros(b, self.num_semantic_classes, 1, 1).to(masks.device)

        # Predict canonical and scale/shift per class
        for i in range(self.num_semantic_classes):

            # Concat mask
            input_feature_map_current = torch.cat((input_feature_map, masks[:, i, :, :].unsqueeze(1)), dim=1)

            # Prob and canonical
            prob = self.p_heads[i](input_feature_map_current)
            depth_rc_current = (prob * bins_map.detach()).sum(1, keepdim=True)
            depth_rc.append(depth_rc_current)
           
            uncertainty_map_current = torch.sqrt((prob * ((bins_map.detach() - depth_rc_current.repeat(1, bin_num+1, 1, 1))**2)).sum(1, keepdim=True))
            uncertainty_map[:, i, :, :] = uncertainty_map_current.squeeze(1)

            # Labels
            pred_label = get_label(torch.squeeze(depth_rc_current, 1), bins_map, bin_num).unsqueeze(1)
            depth_c[:, i, :, : ]  = (torch.gather(bins_map.detach(), 1, pred_label.detach())).squeeze(1)
            
            # Scale/shift

            scale_shift_current = self.s_heads[i](input_feature_map_current)
            if self.virtual_depth_variation == 0:
                pred_scale_shift.append(scale_shift_current)
            elif self.virtual_depth_variation == 1:
                pred_scale_shift.append(scale_shift_current)
            elif self.virtual_depth_variation == 2:
                scale = (scale_shift_current * bins_map_scale.detach()).sum(1, keepdim=True)
                pred_scale_shift.append(scale)

                uncertainty_map_current_scale = torch.sqrt((scale_shift_current * ((bins_map_scale.detach() - scale)**2)).sum(1, keepdim=True))
                uncertainty_map_scale[:, i, :, :] = uncertainty_map_current_scale.unsqueeze(-1)

                pred_label = get_label(torch.squeeze(scale, 1), bins_map_scale, self.bins_scale).unsqueeze(1)
                depth_scale_c = torch.gather(bins_map_scale.detach(), 1, pred_label.detach())
                c_map_scale[:, i, :, :]  = depth_scale_c.unsqueeze(1)     

        pred_depths_c_list.append(depth_c)
        
        uncertainty_maps_list.append(uncertainty_map)
        depth_rc = torch.cat(depth_rc, dim=1) 
        pred_depths_rc_list.append(depth_rc)
        pred_scale_shift = torch.cat(pred_scale_shift, dim=1) 
        
        if self.virtual_depth_variation == 0:
            pred_scale_list.append(pred_scale_shift[:, ::2])  # b, scale
            pred_shift_list.append(pred_scale_shift[:, 1::2]) # b, shift
        elif self.virtual_depth_variation == 1:
            pred_scale_list.append(pred_scale_shift)
        elif self.virtual_depth_variation == 2:
            uncertainty_maps_scale_list.append(uncertainty_map_scale)
            pred_depths_scale_c_list.append(c_map_scale)
            pred_scale_list.append(pred_scale_shift)

          
        # Metric
        if self.loss_type == 0:
            if self.virtual_depth_variation == 0:
                depth_r = (self.relu(depth_rc * pred_scale_list[-1].unsqueeze(-1).unsqueeze(-1) + pred_shift_list[-1].unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
            else:
                depth_r = (self.relu(depth_rc * pred_scale_list[-1].unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
        else:
            if self.virtual_depth_variation == 0:
                depth_r = depth_rc * pred_scale_list[-1].unsqueeze(-1).unsqueeze(-1) + pred_shift_list[-1].unsqueeze(-1).unsqueeze(-1)
            else:
                depth_r = depth_rc * pred_scale_list[-1].unsqueeze(-1).unsqueeze(-1) 

        pred_depths_r_list.append(depth_r)
        
        result = {}
        result["pred_depths_c_list"] = pred_depths_c_list
        result["pred_depths_scale_c_list"] = pred_depths_scale_c_list
        result["uncertainty_maps_list"] = uncertainty_maps_list
        result["uncertainty_maps_scale_list"] = uncertainty_maps_scale_list
        result["pred_depths_r_list"] = pred_depths_r_list
        result["pred_depths_rc_list"] = pred_depths_rc_list
        result["pred_scale_list"] = pred_scale_list
        result["pred_shift_list"] = pred_shift_list

        return result


class RegressionSegmentationNoMasking(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, loss_type=0, num_semantic_classes=5):
        super(RegressionSegmentationNoMasking, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.loss_type = loss_type

        # Each class predicts [0, 1] canonical
        self.sigmoid_head = SigmoidHead(hidden_dim, hidden_dim, num_classes=self.num_semantic_classes) 
        self.s_head = SSHead(hidden_dim, num_out=self.num_semantic_classes * 2)                 

        self.relu = nn.ReLU(inplace=True)

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, masks):
        pred_depths_r_list = []    # Metric 
        pred_depths_rc_list = []   # Canonical

        pred_scale_list = []
        pred_shift_list = []

        b, _, h, w = depth.shape
        
        pred_canonical = self.sigmoid_head(input_feature_map)        
        pred_scale_shift = self.s_head(input_feature_map) 
       
        pred_scale_list.append(pred_scale_shift[:, ::2])  # b, c
        pred_shift_list.append(pred_scale_shift[:, 1::2]) # b, c
          
        # Canonical
        depth_rc = pred_canonical
        pred_depths_rc_list.append(depth_rc)
        
        # Metric
        if self.loss_type == 0:
            depth_r = (self.relu(depth_rc * pred_scale_shift[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale_shift[:, 1::2].unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
        else:
            depth_r = depth_rc * pred_scale_shift[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale_shift[:, 1::2].unsqueeze(-1).unsqueeze(-1)
        
        # depth_r: b, c, h, w
        pred_depths_r_list.append(depth_r)
        
        result = {}
        result["pred_depths_r_list"] = pred_depths_r_list
        result["pred_depths_rc_list"] = pred_depths_rc_list
        result["pred_scale_list"] = pred_scale_list
        result["pred_shift_list"] = pred_shift_list

        return result


class RegressionSegmentationNoMaskingConcatMasks(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, loss_type=0, num_semantic_classes=14):
        super(RegressionSegmentationNoMaskingConcatMasks, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.loss_type = loss_type

        self.sigmoid_head = SigmoidHead(hidden_dim + num_semantic_classes, hidden_dim, num_classes=self.num_semantic_classes) 
        self.s_head = SSHead(hidden_dim + num_semantic_classes, num_out=self.num_semantic_classes * 2)                  

        self.relu = nn.ReLU(inplace=True)

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, masks):
        pred_depths_r_list = []    # Metric 
        pred_depths_rc_list = []   # Canonical

        pred_scale_list = []
        pred_shift_list = []

        b, _, h, w = depth.shape
        
        input_feature_map = torch.cat((input_feature_map, masks),dim=1)
        pred_prob = self.sigmoid_head(input_feature_map)        
        pred_scale_shift = self.s_head(input_feature_map)       
       
        pred_scale_list.append(pred_scale_shift[:, ::2])  # b, c
        pred_shift_list.append(pred_scale_shift[:, 1::2]) # b, c
          
        # Canonical
        depth_rc = pred_prob
        pred_depths_rc_list.append(depth_rc)
        
        # Metric
        if self.loss_type == 0:
            depth_r = (self.relu(depth_rc * pred_scale_shift[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale_shift[:, 1::2].unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
        else:
            depth_r = depth_rc * pred_scale_shift[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale_shift[:, 1::2].unsqueeze(-1).unsqueeze(-1)
        
        pred_depths_r_list.append(depth_r)
        
        result = {}
        result["pred_depths_r_list"] = pred_depths_r_list
        result["pred_depths_rc_list"] = pred_depths_rc_list
        result["pred_scale_list"] = pred_scale_list
        result["pred_shift_list"] = pred_shift_list

        return result


class RegressionSegmentationModuleListConcatMasks(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, loss_type=0, num_semantic_classes=5):
        super(RegressionSegmentationModuleListConcatMasks, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.loss_type = loss_type

        self.sigmoid_heads = nn.ModuleList()
        self.s_heads = nn.ModuleList()

        for i in range(num_semantic_classes):       
            self.sigmoid_heads.append(SigmoidHead(128 + 1, 128, num_classes=1))
            self.s_heads.append(SSHead(128 + 1, num_out=1 * 2))                

        self.relu = nn.ReLU(inplace=True)

    def forward(self, depth, context, input_feature_map, min_depth, max_depth, masks):
        pred_depths_r_list = []    # Metric 
        pred_depths_rc_list = []   # Canonical
        pred_scale_list = []
        pred_shift_list = []

        b, _, h, w = depth.shape

        pred_canonical = []
        pred_scale_shift = [] 
        for i in range(self.num_semantic_classes):
            # Concat masks
            input_feature_map_current = torch.cat((input_feature_map, masks[:, i, :, :].unsqueeze(1)), dim=1)

            canonical = self.sigmoid_heads[i](input_feature_map_current)
            scale_shift_current = self.s_heads[i](input_feature_map_current)

            pred_canonical.append(canonical)
            pred_scale_shift.append(scale_shift_current)

        pred_canonical = torch.cat(pred_canonical, dim=1) 
        pred_scale_shift = torch.cat(pred_scale_shift, dim=1) 

        pred_scale_list.append(pred_scale_shift[:, ::2])  # b, c
        pred_shift_list.append(pred_scale_shift[:, 1::2]) # b, c
          
        # Canonical
        depth_rc = pred_canonical
        pred_depths_rc_list.append(depth_rc)
        
        # Metric
        if self.loss_type == 0:
            depth_r = (self.relu(depth_rc * pred_scale_shift[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale_shift[:, 1::2].unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
        else:
            depth_r = depth_rc * pred_scale_shift[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale_shift[:, 1::2].unsqueeze(-1).unsqueeze(-1)
        
        # depth_r: b, c, h, w
        pred_depths_r_list.append(depth_r)
        
        result = {}
        result["pred_depths_r_list"] = pred_depths_r_list
        result["pred_depths_rc_list"] = pred_depths_rc_list
        result["pred_scale_list"] = pred_scale_list
        result["pred_shift_list"] = pred_shift_list

        return result


"""
Instances variations 
"""


class UniformInstancesSharedCanonical(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=14, \
                 num_instances=63,var=0, padding_instances=0, roi_align=0, roi_align_size=32, bins_scale=150, virtual_depth_variation=0):
        super(UniformInstancesSharedCanonical, self).__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.loss_type = loss_type
        self.bins_scale = bins_scale

        self.num_semantic_classes = num_semantic_classes
        self.num_instances = num_instances
        self.var = var # Variation 
        self.virtual_depth_variation = virtual_depth_variation

        global padding_global
        
        self.padding_instances = padding_instances
        padding_global = padding_instances
     

        if self.virtual_depth_variation == 0:
            self.instances_scale_and_shift = ROISelectScale(128, downsampling=4, num_out=2*(self.num_semantic_classes-1)) # Do not include null 
        elif self.virtual_depth_variation == 1:
            self.instances_scale_and_shift = ROISelectScale(128, downsampling=4, num_out=self.num_semantic_classes-1) # Do not include null 
        elif self.virtual_depth_variation == 2:
            self.instances_scale_and_shift = ROISelectScaleBins(128, downsampling=4, num_semantic_classes= self.num_semantic_classes - 1, num_bins=self.bins_scale) # Do not include null 
 
        self.instances_canonical = ROISelectSharedCanonicalUniform(128, bin_num=bin_num+1) 

        self.roi_align = roi_align 
        if roi_align == 1:
            output_size = (roi_align_size, roi_align_size)
            spatial_scale  = 1.0/4
            sampling_ratio = 4

            self.roiAlign = RoIAlign(output_size, spatial_scale=spatial_scale, sampling_ratio=sampling_ratio)
        
        # Pick variation #      
        if var == 1:
            self.instances_canonical = ROISelectSharedCanonicalBigUniform(128, bin_num=bin_num+1)          
        if var == 2:
            self.instances_canonical = ROISelectSharedCanonicalHugeUniform(128, bin_num=bin_num+1)       
        if var == 3:
            if self.virtual_depth_variation == 0:
                self.instances_scale_and_shift = ROISelectScaleBig(128, downsampling=4, num_out=2*(self.num_semantic_classes-1)) # Do not include null 
            elif self.virtual_depth_variation == 1:
                self.instances_scale_and_shift = ROISelectScaleBig(128, downsampling=4, num_out=self.num_semantic_classes-1) # Do not include null 
            elif self.virtual_depth_variation == 2:
                self.instances_scale_and_shift = ROISelectScaleBigBins(128, downsampling=4, num_semantic_classes= self.num_semantic_classes - 1, num_bins=self.bins_scale) # Do not include null 
        if var == 4:
            if self.virtual_depth_variation == 0:
                self.instances_scale_and_shift = ROISelectScaleHuge(128, downsampling=4, num_out=2*(self.num_semantic_classes-1)) # Do not include null 
            elif self.virtual_depth_variation == 1:
                self.instances_scale_and_shift = ROISelectScaleHuge(128, downsampling=4, num_out=self.num_semantic_classes-1) # Do not include null 
            elif self.virtual_depth_variation == 2:
                self.instances_scale_and_shift = ROISelectScaleHugeBins(128, downsampling=4, num_semantic_classes= self.num_semantic_classes - 1, num_bins=self.bins_scale) # Do not include null 
        if var == 5:
            if self.virtual_depth_variation == 0:
                self.instances_scale_and_shift = ROISelectScaleSmall(128, downsampling=4, num_out=2*(self.num_semantic_classes-1)) # Do not include null 
            elif self.virtual_depth_variation == 1:
                self.instances_scale_and_shift = ROISelectScaleSmall(128, downsampling=4, num_out=self.num_semantic_classes-1) # Do not include null 
            elif self.virtual_depth_variation == 2:
                self.instances_scale_and_shift = ROISelectScaleSmallBins(128, downsampling=4, num_semantic_classes= self.num_semantic_classes - 1, num_bins=self.bins_scale) # Do not include null 

        self.relu = nn.ReLU(inplace=True)

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, masks, instances, boxes, labels):
        pred_depths_instances_r_list = [] # Metric
        pred_depths_instances_rc_list = [] # Canonical

        pred_scale_instances_list = []
        pred_shift_instances_list = []

        uncertainty_maps_list = []
        uncertainty_maps_scale_list = []
        pred_depths_c_list = [] 

        batch_size, _, h, w = depth.shape
        max_instances_size = instances.shape[1]

        # Mask feature maps based on bboxes #
        if self.roi_align == 0:
            input_feature_map_instances_roi = roi_select_features(input_feature_map, boxes, labels) 
        else:
            boxes_roi, _ = get_valid_boxes(boxes, labels)
            instances_per_batch, num_valid_boxes = get_valid_num_instances_per_batch(boxes, labels)
            instances_per_batch = torch.cat((torch.tensor([0]).to(labels.device), instances_per_batch))
            instances_per_batch = torch.cumsum(instances_per_batch, dim=0)
           
            boxes_roi = boxes_roi.to(torch.float32)
            boxes_roi -= 0.5
            boxes_roi = [(boxes_roi[instances_per_batch[i]:instances_per_batch[i+1]]) for i in range(instances_per_batch.shape[0] - 1)]
            
            input_feature_map_instances_roi = self.roiAlign(input_feature_map, boxes_roi) 

        if self.virtual_depth_variation == 0:
            instances_scale_shift = self.instances_scale_and_shift(input_feature_map_instances_roi, boxes, labels)
            instances_scale, instances_shift = pick_predictions_instances_scale_shift(instances_scale_shift, labels)
            pred_scale_instances_list.append(instances_scale)
            pred_shift_instances_list.append(instances_shift)
        elif self.virtual_depth_variation == 1:
            instances_scale_shift = self.instances_scale_and_shift(input_feature_map_instances_roi, boxes, labels)
            instances_scale, instances_shift = pick_predictions_instances_scale_shift(instances_scale_shift, labels, only_scale=True)
            pred_scale_instances_list.append(instances_scale)
        elif self.virtual_depth_variation == 2:
            instances_scale_unc = self.instances_scale_and_shift(input_feature_map_instances_roi, boxes, labels)
            instances_scale, instances_scale_unc = pick_predictions_instances_scale_shift(instances_scale_unc, labels)
            pred_scale_instances_list.append(instances_scale)
            uncertainty_maps_scale_list.append(instances_scale_unc)
 
        # Canonical instances valid #
        instances_canonical_prob = self.instances_canonical(input_feature_map, boxes, labels)
       
        bins_map = get_uniform_bins(depth, min_depth, max_depth, bin_num)[0, :, :, :].unsqueeze(0)
        bins_map = torch.cat([bins_map] * instances_canonical_prob.shape[0], dim=0)  
        depth_instances_rc = (instances_canonical_prob * bins_map.detach()).sum(1, keepdim=True)
      
        # Unc #
        uncertainty_map_current = torch.sqrt((instances_canonical_prob * ((bins_map.detach() - depth_instances_rc.repeat(1, bin_num+1, 1, 1))**2)).sum(1, keepdim=True))
        uncertainty_map = uncertainty_map_current
        uncertainty_maps_list.append(uncertainty_map)

        # Labels #
        pred_label = get_label(torch.squeeze(depth_instances_rc, 1), bins_map, bin_num).unsqueeze(1)
        depth_c  = (torch.gather(bins_map.detach(), 1, pred_label.detach()))
        pred_depths_c_list.append(depth_c)

        # Fill full instances map canonical #        
        canonical_full = torch.zeros((batch_size*max_instances_size, 1, h, w)).to(depth_instances_rc.device)
        
        valid_boxes = labels.view(batch_size * max_instances_size, 1)
        valid_boxes = torch.nonzero(valid_boxes != 0)

        canonical_full[valid_boxes[:,0]] = depth_instances_rc
        depth_instances_rc = canonical_full.view(batch_size, max_instances_size, h, w)
        pred_depths_instances_rc_list.append(depth_instances_rc)
   
        # Metric
        if self.loss_type == 0:
            if self.virtual_depth_variation == 0:
                depth_instances_r = (self.relu(depth_instances_rc * instances_scale.unsqueeze(-1).unsqueeze(-1) + instances_shift.unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
            else:
                depth_instances_r = (self.relu(depth_instances_rc * instances_scale.unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
        else:
            if self.virtual_depth_variation == 0:
                depth_instances_r = depth_instances_rc * instances_scale.unsqueeze(-1).unsqueeze(-1) + instances_shift.unsqueeze(-1).unsqueeze(-1)
            else:
                depth_instances_r = depth_instances_rc * instances_scale.unsqueeze(-1).unsqueeze(-1)

        # depth_r: b, c, h, w
        pred_depths_instances_r_list.append(depth_instances_r)
        result = {}
        result["pred_depths_c_list"] = pred_depths_c_list
        result["uncertainty_maps_list"] = uncertainty_maps_list
        result["uncertainty_maps_scale_list"] = uncertainty_maps_scale_list

        result["pred_depths_instances_r_list"] = pred_depths_instances_r_list
        result["pred_depths_instances_rc_list"] = pred_depths_instances_rc_list
        result["pred_scale_instances_list"] = pred_scale_instances_list
        result["pred_shift_instances_list"] = pred_shift_instances_list

        return result


class RegressionInstances(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, loss_type=0, \
                 num_semantic_classes=14, num_instances=63,var=0,padding_instances=0):
        super(RegressionInstances, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.loss_type = loss_type
        
        self.num_instances = num_instances
        self.var = var
        
        global padding_global
        self.padding_instances = padding_instances
        padding_global = padding_instances
        
  
        self.instances_scale_and_shift = ROISelectScale(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
        self.instances_canonical = ROISelectCanonical(128, 4, num_semantic_classes=self.num_semantic_classes-1)      

        # Pick variation # 
        if var == 1:
            self.instances_scale_and_shift = ROISelectScaleA(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
            self.instances_canonical = ROISelectCanonical(128, 4, num_semantic_classes=self.num_semantic_classes-1)       
        if var == 2:
            self.instances_scale_and_shift = ROISelectScaleB(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
            self.instances_canonical = ROISelectCanonical(128, 4, num_semantic_classes=self.num_semantic_classes-1)       
        if var == 3:
            self.instances_scale_and_shift = ROISelectScale(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
            self.instances_canonical = ROISelectCanonicalA(128, num_semantic_classes=self.num_semantic_classes-1)   
        if var == 4:
            self.instances_scale_and_shift = ROISelectScale(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
            self.instances_canonical = ROISelectCanonicalB(128, num_semantic_classes=self.num_semantic_classes-1)   
        if var == 5:
            self.instances_scale_and_shift = ROISelectScale(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
            self.instances_canonical = ROISelectCanonicalC(128, num_semantic_classes=self.num_semantic_classes-1)   
        if var == 6:
            self.instances_scale_and_shift = ROISelectScaleB(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
            self.instances_canonical = ROISelectCanonicalC(128, num_semantic_classes=self.num_semantic_classes-1)          
        if var == 7:
            self.instances_scale_and_shift = ROISelectScaleA(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
            self.instances_canonical = ROISelectCanonicalC(128, num_semantic_classes=self.num_semantic_classes-1) 
        if var == 8:
            self.instances_scale_and_shift = ROISelectScaleC(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
            self.instances_canonical = ROISelectCanonical(128, 4, num_semantic_classes=self.num_semantic_classes-1) 
        if var == 9:
            self.instances_scale_and_shift = ROISelectScale(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
            self.instances_canonical = ROISelectCanonicalD(128, num_semantic_classes=self.num_semantic_classes-1) 
        if var == 10:
            self.instances_scale_and_shift = ROISelectScaleC(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
            self.instances_canonical = ROISelectCanonicalD(128, num_semantic_classes=self.num_semantic_classes-1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, masks, instances, boxes, labels):
        pred_depths_instances_r_list = [] # Metric 
        pred_depths_instances_rc_list = [] # Canonical

        pred_scale_instances_list = []
        pred_shift_instances_list = []

        # Mask feature maps based on bboxes #
        input_feature_map_instances_roi = roi_select_features(input_feature_map, boxes, labels) 

        # Scale/Shift #
        instances_scale_shift = self.instances_scale_and_shift(input_feature_map_instances_roi, boxes, labels)
        instances_scale, instances_shift = pick_predictions_instances_scale_shift(instances_scale_shift, labels)
        pred_scale_instances_list.append(instances_scale)
        pred_shift_instances_list.append(instances_shift)

        # Canonical #
        instances_canonical = self.instances_canonical(input_feature_map_instances_roi, boxes, labels)
        instances_canonical = pick_predictions_instances_canonical(instances_canonical, labels)
        pred_depths_instances_rc_list.append(instances_canonical)
        
        # Metric
        if self.loss_type == 0:
            depth_instances_r = (self.relu(instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) + instances_shift.unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
        else:
            depth_instances_r = instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) + instances_shift.unsqueeze(-1).unsqueeze(-1)
       
        pred_depths_instances_r_list.append(depth_instances_r)
        
        result = {}
        result["pred_depths_instances_r_list"] = pred_depths_instances_r_list
        result["pred_depths_instances_rc_list"] = pred_depths_instances_rc_list
        result["pred_scale_instances_list"] = pred_scale_instances_list
        result["pred_shift_instances_list"] = pred_shift_instances_list

        return result


class RegressionInstancesSharedCanonical(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, loss_type=0, num_semantic_classes=14, num_instances=63,var=0, padding_instances=0):
        super(RegressionInstancesSharedCanonical, self).__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.loss_type = loss_type
        
        self.num_semantic_classes = num_semantic_classes
        self.num_instances = num_instances
        self.var = var
        
        global padding
        
        self.padding_instances = padding_instances
        padding_global = padding_instances
      
        self.instances_scale_and_shift = ROISelectScale(128, downsampling=4, num_out=2*(self.num_semantic_classes-1))
        self.instances_canonical = ROISelectSharedCanonical(128, num_semantic_classes=1)       
        
        # Pick variation #      
        if var == 1:
            self.instances_canonical = ROISelectSharedCanonicalBig(128, num_semantic_classes=1)          
        if var == 2:
            self.instances_canonical = ROISelectSharedCanonicalHuge(128, num_semantic_classes=1)       
        if var == 3:
            self.instances_scale_and_shift = ROISelectScaleBig(128, downsampling=4, num_out=2*(self.num_semantic_classes-1))
        if var == 4:
            self.instances_scale_and_shift = ROISelectScaleHuge(128, downsampling=4, num_out=2*(self.num_semantic_classes-1))
        if var == 5:
            self.instances_scale_and_shift = ROISelectScaleSmall(128, downsampling=4, num_out=2*(self.num_semantic_classes-1))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, masks, instances, boxes, labels):
        pred_depths_instances_r_list = []  # Metric
        pred_depths_instances_rc_list = [] # Canonical

        pred_scale_instances_list = []
        pred_shift_instances_list = []

        batch_size, num_max_instances, h, w = instances.shape

        # Mask feature maps based on bboxes #
        input_feature_map_instances_roi = roi_select_features(input_feature_map, boxes, labels) 
        
        # Scale/Shift #
        instances_scale_shift = self.instances_scale_and_shift(input_feature_map_instances_roi, boxes, labels)
        instances_scale, instances_shift = pick_predictions_instances_scale_shift(instances_scale_shift, labels)
        pred_scale_instances_list.append(instances_scale)
        pred_shift_instances_list.append(instances_shift)

        # Shared canonical #
        instances_canonical = self.instances_canonical(input_feature_map, boxes, labels)
        
        boxes_valid_idx = get_valid_boxes_idx(boxes, labels)

        canonical_full = torch.zeros((batch_size*num_max_instances, 1, h, w)).to(instances_canonical.device)
        canonical_full[boxes_valid_idx[:,0]] = instances_canonical 

        canonical_full = canonical_full.view(batch_size, num_max_instances, h, w)
        instances_canonical = canonical_full
        pred_depths_instances_rc_list.append(instances_canonical)
    
        # Metric
        if self.loss_type == 0:
            depth_instances_r = (self.relu(instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) + instances_shift.unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
        else:
            if self.var == 8:
                depth_instances_r = instances_canonical * 50 * F.sigmoid(instances_scale.unsqueeze(-1).unsqueeze(-1)) + instances_shift.unsqueeze(-1).unsqueeze(-1)
            
            elif self.var == 9 or self.var == 13 or self.var == 14 or self.var == 15:
                depth_instances_r = instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) + instances_shift.unsqueeze(-1).unsqueeze(-1)
            
            else:
                depth_instances_r = instances_canonical * 20 * F.sigmoid(instances_scale.unsqueeze(-1).unsqueeze(-1)) + instances_shift.unsqueeze(-1).unsqueeze(-1)
       
        pred_depths_instances_r_list.append(depth_instances_r)
        
        result = {}
        result["pred_depths_instances_r_list"] = pred_depths_instances_r_list
        result["pred_depths_instances_rc_list"] = pred_depths_instances_rc_list
        result["pred_scale_instances_list"] = pred_scale_instances_list
        result["pred_shift_instances_list"] = pred_shift_instances_list

        return result


class RegressionInstancesSharedCanonicalClass(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, loss_type=0, num_semantic_classes=14, num_instances=63,var=0, padding_instances=0):
        super(RegressionInstancesSharedCanonicalClass, self).__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.loss_type = loss_type
        
        self.num_semantic_classes = num_semantic_classes
        self.num_instances = num_instances
        self.var = var

        global padding
        
        self.padding_instances = padding_instances
        padding_global = padding_instances
      
        self.instances_scale_and_shift = ROISelectScale(128, downsampling=4, num_out=2*(self.num_semantic_classes-1))
        self.instances_canonical = ROISelectCanonicalSharedClass(128, num_semantic_classes=self.num_semantic_classes-1)
        
        # Pick variation #      
        if var == 1 or var == 3:
            self.instances_scale_and_shift = ROISelectScaleBig(128, downsampling=4, num_out=2*(self.num_semantic_classes-1))
        if var == 2 or var == 3:
            self.instances_canonical = ROISelectCanonicalSharedClassBig(128, num_semantic_classes=self.num_semantic_classes-1)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, masks, instances, boxes, labels):
        pred_depths_instances_r_list = []
        pred_depths_instances_rc_list = []

        pred_scale_instances_list = []
        pred_shift_instances_list = []

        input_feature_map_instances = input_feature_map
        batch_size, hidden_dim, height_hidden, width_hidden = input_feature_map_instances.shape

        # Mask feature maps based on bboxes #
        input_feature_map_instances_roi = roi_select_features(input_feature_map, boxes, labels)

        # Scale/shift #
        instances_scale_shift = self.instances_scale_and_shift(input_feature_map_instances_roi, boxes, labels)
        instances_scale, instances_shift = pick_predictions_instances_scale_shift(instances_scale_shift, labels)
        pred_scale_instances_list.append(instances_scale)
        pred_shift_instances_list.append(instances_shift)

        # Shared canonical per class #
        input_feature_map_instances_roi_canonical = roi_select_features_canonical_shared(input_feature_map_instances, boxes, labels, num_semantic_classes=self.num_semantic_classes)
        
        input_feature_map_instances_roi_canonical = input_feature_map_instances_roi_canonical.view(batch_size*(self.num_semantic_classes-1), hidden_dim, height_hidden, width_hidden)
        instances_canonical = self.instances_canonical(input_feature_map_instances_roi_canonical)
        
        instances_canonical = pick_predictions_instances_canonical_shared_class(instances_canonical, labels)
        pred_depths_instances_rc_list.append(instances_canonical)
    
        # Metric
        if self.loss_type == 0:
            depth_instances_r = (self.relu(instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) + instances_shift.unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
        else:
            if self.var == 8:
                depth_instances_r = instances_canonical * 50 * F.sigmoid(instances_scale.unsqueeze(-1).unsqueeze(-1)) + instances_shift.unsqueeze(-1).unsqueeze(-1)
            
            elif self.var == 9 or self.var == 13 or self.var == 14 or self.var == 15:
                depth_instances_r = instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) + instances_shift.unsqueeze(-1).unsqueeze(-1)
            
            else:
                depth_instances_r = instances_canonical * 20 * F.sigmoid(instances_scale.unsqueeze(-1).unsqueeze(-1)) + instances_shift.unsqueeze(-1).unsqueeze(-1)
       
        pred_depths_instances_r_list.append(depth_instances_r)
        
        result = {}
        result["pred_depths_instances_r_list"] = pred_depths_instances_r_list
        result["pred_depths_instances_rc_list"] = pred_depths_instances_rc_list
        result["pred_scale_instances_list"] = pred_scale_instances_list
        result["pred_shift_instances_list"] = pred_shift_instances_list

        return result


class RegressionInstancesSharedCanonicalModuleScale(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, loss_type=0, num_semantic_classes=14, num_instances=63,var=0, padding_instances=0):
        super(RegressionInstancesSharedCanonicalModuleScale, self).__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.loss_type = loss_type

        self.num_semantic_classes = num_semantic_classes
        self.num_instances = num_instances
        self.var = var

        global padding
        
        self.padding_instances = padding_instances
        padding_global = padding_instances
      
        self.instances_scale_and_shift = nn.ModuleList()
        for i in range(num_semantic_classes - 1):
            self.instances_scale_and_shift.append(ROISelectScaleModule(128, downsampling=4, num_semantic_classes=1))

        self.instances_canonical = ROISelectSharedCanonical(128, num_semantic_classes=1)      

        # Pick variation #      
        if var == 1 or var == 3:
            for i in range(num_semantic_classes - 1):
                self.instances_scale_and_shift.append(ROISelectScaleModuleBig(128, downsampling=4, num_semantic_classes=1))        
        if var == 2 or var == 3:       
             self.instances_canonical = ROISelectSharedCanonicalBig(128, num_semantic_classes=1)    
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, masks, instances, boxes, labels):
        pred_depths_instances_r_list = []  # Metric 
        pred_depths_instances_rc_list = [] # Canonical

        pred_scale_instances_list = []
        pred_shift_instances_list = []

        batch_size, num_max_instances, h, w = instances.shape

        labels_reshaped = labels.view(batch_size * num_max_instances, 1)

        instances_scale = torch.zeros((batch_size*num_max_instances, 1)).to(labels.device) 
        instances_shift = torch.zeros((batch_size*num_max_instances, 1)).to(labels.device)

        # Pass class-specific feature maps to appropriate scale head # 
        for i in range(self.num_semantic_classes-1):
            with torch.no_grad():
                valid_boxes_class = torch.nonzero(labels_reshaped == i + 1)

                if valid_boxes_class.shape[0] == 0:
                    continue

            input_feature_map_instances_roi = roi_select_features_module(input_feature_map, boxes, labels, i+1)   
            scale_shift = self.instances_scale_and_shift[i](input_feature_map_instances_roi, boxes, labels, i+1)
            
            instances_scale[valid_boxes_class[:, 0]] = scale_shift[:, ::2]
            instances_shift[valid_boxes_class[:, 0]] = scale_shift[:, 1::2]

        instances_scale = instances_scale.view(batch_size, num_max_instances)
        instances_shift = instances_shift.view(batch_size, num_max_instances)
        
        pred_scale_instances_list.append(instances_scale)
        pred_shift_instances_list.append(instances_shift)

        # Canonical shared #
        instances_canonical = self.instances_canonical(input_feature_map, boxes, labels)
         
        # Copy the shared canonical to all instances #
        valid_boxes = torch.nonzero(labels_reshaped != 0)
        canonical_full = torch.zeros((batch_size*num_max_instances, 1, h, w)).to(instances_canonical.device)
        
        canonical_full[valid_boxes[:,0]] = instances_canonical 
        canonical_full = canonical_full.view(batch_size, num_max_instances, h, w)
        instances_canonical = canonical_full
        pred_depths_instances_rc_list.append(instances_canonical)
    
        # Metric
        if self.loss_type == 0:
            depth_instances_r = (self.relu(instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) + instances_shift.unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
        else:
            depth_instances_r = instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) + instances_shift.unsqueeze(-1).unsqueeze(-1)
       
        pred_depths_instances_r_list.append(depth_instances_r)
        
        result = {}

        result["pred_depths_instances_r_list"] = pred_depths_instances_r_list
        result["pred_depths_instances_rc_list"] = pred_depths_instances_rc_list
        result["pred_scale_instances_list"] = pred_scale_instances_list
        result["pred_shift_instances_list"] = pred_shift_instances_list

        return result


class RegressionInstancesModule(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, loss_type=0, num_semantic_classes=14, num_instances=63,var=0, padding_instances=0):
        super(RegressionInstancesModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.loss_type = loss_type
        
        self.num_semantic_classes = num_semantic_classes
        self.num_instances = num_instances
        self.var = var

        global padding
        
        self.padding_instances = padding_instances
        padding_global = padding_instances
      
        self.instances_scale_and_shift = nn.ModuleList()
        for i in range(num_semantic_classes - 1):
            self.instances_scale_and_shift.append(ROISelectScaleModule(128, downsampling=4, num_semantic_classes=1))

        self.instances_canonical = nn.ModuleList()
        for i in range(num_semantic_classes - 1):
            self.instances_canonical.append(ROISelectCanonicalModule(128, num_semantic_classes=1))     
        
        # Pick variation #      
        if var == 1 or var == 3:
            for i in range(num_semantic_classes - 1):
                self.instances_scale_and_shift.append(ROISelectScaleModuleBig(128, downsampling=4, num_semantic_classes=1))        
        if var == 2 or var == 3:
            for i in range(num_semantic_classes - 1):
                self.instances_canonical.append(ROISelectCanonicalModuleBig(128, num_semantic_classes=1))     

        self.relu = nn.ReLU(inplace=True)

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, masks, instances, boxes, labels):
        pred_depths_instances_r_list = []  # Metric 
        pred_depths_instances_rc_list = [] # Canonical

        pred_scale_instances_list = []
        pred_shift_instances_list = []
        b, _, h, w = depth.shape

        batch_size, num_max_instances, h, w = instances.shape

        labels_reshaped = labels.view(batch_size * num_max_instances, 1)
        
        instances_scale = torch.zeros((batch_size*num_max_instances, 1)).to(labels.device) 
        instances_shift = torch.zeros((batch_size*num_max_instances, 1)).to(labels.device)
        canonical_full = torch.zeros((batch_size*num_max_instances, 1, h, w)).to(labels.device)
 
        # Pass class-specific feature maps to appropriate scale and canonical heads # 
        for i in range(self.num_semantic_classes-1):
            with torch.no_grad():
                valid_boxes_class = torch.nonzero(labels_reshaped == i + 1)

                if valid_boxes_class.shape[0] == 0:
                    continue

            # Scale/Shift #
            input_feature_map_instances_roi = roi_select_features_module(input_feature_map, boxes, labels, i+1)   
            scale_shift = self.instances_scale_and_shift[i](input_feature_map_instances_roi, boxes, labels, i+1)
            instances_scale[valid_boxes_class[:, 0]] = scale_shift[:, ::2]
            instances_shift[valid_boxes_class[:, 0]] = scale_shift[:, 1::2]

            # Canonical #
            instances_canonical = self.instances_canonical[i](input_feature_map_instances_roi, boxes, labels, i+1)
            canonical_full[valid_boxes_class[:, 0]] = instances_canonical

        # Scale/Shift #
        instances_scale = instances_scale.view(batch_size, num_max_instances)
        instances_shift = instances_shift.view(batch_size, num_max_instances)
        pred_scale_instances_list.append(instances_scale)
        pred_shift_instances_list.append(instances_shift)
        
        # Canonical #
        instances_canonical = canonical_full.view(batch_size, num_max_instances, h, w)
        pred_depths_instances_rc_list.append(instances_canonical)
    
        # Metric
        if self.loss_type == 0:
            depth_instances_r = (self.relu(instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) + instances_shift.unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
        else:
            depth_instances_r = instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) + instances_shift.unsqueeze(-1).unsqueeze(-1)
       
        pred_depths_instances_r_list.append(depth_instances_r)
        
        result = {}
        result["pred_depths_instances_r_list"] = pred_depths_instances_r_list
        result["pred_depths_instances_rc_list"] = pred_depths_instances_rc_list
        result["pred_scale_instances_list"] = pred_scale_instances_list
        result["pred_shift_instances_list"] = pred_shift_instances_list

        return result
