import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

padding_global = 0

from .utils_clean import *


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
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0):
        super(Regression, self).__init__()

        self.r_head = RegressionHead(hidden_dim, hidden_dim) # Canonical regression [0, 1]

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth):
        pred_depths_r_list = []    # Metric 
         
        depth_r = self.r_head(input_feature_map)        
        
        # Metric
        if self.loss_type == 0:
            depth_r = self.relu(depth_rc).clamp(min=1e-3)

        pred_depths_r_list.append(depth_r)

        result = {}
        result["pred_depths_r_list"] = pred_depths_r_list

        return result


"""
Regression prediction with single scale per image 
"""
class RegressionSingleScale(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0):
        super(RegressionSingleScale, self).__init__()

        self.sigmoid_head = SigmoidHead(hidden_dim, hidden_dim) # Canonical regression [0, 1]
        self.s_head = SSHead(hidden_dim)            # Global scale and shift 

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
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0):
        super(UniformSingleScale, self).__init__()
        self.loss_type = loss_type

        self.p_head = PHead(hidden_dim, hidden_dim, bin_num + 1) # Propabilities canonical
        self.s_head = SSHead(hidden_dim)                         # Global scale and shift 

        self.relu = nn.ReLU(inplace=True)

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth):
        pred_depths_r_list = []    # Metric 
        pred_depths_rc_list = []   # Canonical

        bins_map = get_uniform_bins(depth, min_depth, max_depth, bin_num)

        pred_scale_list = []
        pred_shift_list = []
        uncertainty_maps_list = []
        pred_depths_c_list = [] 
        
        # Canonical propabilities 
        pred_rc = self.p_head(input_feature_map)       
        
        # Scale and shift 
        pred_scale_shift = self.s_head(input_feature_map)       
        pred_scale_list.append(pred_scale_shift[:, 0:1])
        pred_shift_list.append(pred_scale_shift[:, 1:2])
       
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
            depth_r = (self.relu(depth_rc * pred_scale_shift[:, 0:1].unsqueeze(1).unsqueeze(1) + pred_scale_shift[:, 1:2].unsqueeze(1).unsqueeze(1))).clamp(min=1e-3)
        else:
            depth_r = depth_rc * pred_scale_shift[:, 0:1].unsqueeze(1).unsqueeze(1) + pred_scale_shift[:, 1:2].unsqueeze(1).unsqueeze(1)

        pred_depths_r_list.append(depth_r)

        result = {}
        result["pred_depths_r_list"] = pred_depths_r_list
        result["pred_depths_rc_list"] = pred_depths_rc_list
        result["pred_depths_c_list"] = pred_depths_c_list
        result["uncertainty_maps_list"] = uncertainty_maps_list
        result["pred_scale_list"] = pred_scale_list
        result["pred_shift_list"] = pred_shift_list

        return result

"""
Segmentation variations
"""

class UniformSegmentationModuleListConcatMasks(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=5):
        super(UniformSegmentationModuleListConcatMasks, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.loss_type = loss_type
        
        self.relu = nn.ReLU(inplace=True)

        self.p_heads = nn.ModuleList()
        self.s_heads = nn.ModuleList()

        for i in range(num_semantic_classes):       
            self.p_heads.append(PHead(128 + 1, 128, bin_num=bin_num+1))
            self.s_heads.append(SSHead(128 + 1, num_classes=1))                

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, masks):
        pred_depths_r_list = []    # Metric 
        pred_depths_rc_list = []   # Canonical

        pred_scale_list = []
        pred_shift_list = []

        uncertainty_maps_list = []
        pred_depths_c_list = [] 

        b, _, h, w = depth.shape
        depth_rc = []
        pred_scale_shift = [] 

        bins_map = get_uniform_bins(depth, min_depth, max_depth, bin_num)

        uncertainty_map = torch.zeros(b, self.num_semantic_classes, h, w).to(masks.device)
        depth_c = torch.zeros(b, self.num_semantic_classes, h, w).to(masks.device)

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
            pred_scale_shift.append(scale_shift_current)

        pred_depths_c_list.append(depth_c)
        uncertainty_maps_list.append(uncertainty_map)

        depth_rc = torch.cat(depth_rc, dim=1) 
        pred_depths_rc_list.append(depth_rc)
        
        pred_scale_shift = torch.cat(pred_scale_shift, dim=1) 
        pred_scale_list.append(pred_scale_shift[:, ::2])  # b, c
        pred_shift_list.append(pred_scale_shift[:, 1::2]) # b, c
          
        # Metric
        if self.loss_type == 0:
            depth_r = (self.relu(depth_rc * pred_scale_shift[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale_shift[:, 1::2].unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
        else:
            depth_r = depth_rc * pred_scale_shift[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale_shift[:, 1::2].unsqueeze(-1).unsqueeze(-1)
        
        pred_depths_r_list.append(depth_r)
        
        result = {}
        result["pred_depths_c_list"] = pred_depths_c_list
        result["uncertainty_maps_list"] = uncertainty_maps_list
        result["pred_depths_r_list"] = pred_depths_r_list
        result["pred_depths_rc_list"] = pred_depths_rc_list
        result["pred_scale_list"] = pred_scale_list
        result["pred_shift_list"] = pred_shift_list

        return result


class RegressionSegmentationNoMasking(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=5):
        super(RegressionSegmentationNoMasking, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.loss_type = loss_type

        # Each class predicts [0, 1] canonical
        self.sigmoid_head = SigmoidHead(hidden_dim, hidden_dim, num_classes=self.num_semantic_classes) 
        self.s_head = SSHead(hidden_dim, num_classes=self.num_semantic_classes)                 

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
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=14):
        super(RegressionSegmentationNoMaskingConcatMasks, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.loss_type = loss_type

        self.sigmoid_head = SigmoidHead(hidden_dim + num_semantic_classes, hidden_dim, num_classes=self.num_semantic_classes) 
        self.s_head = SSHead(hidden_dim + num_semantic_classes, num_classes=self.num_semantic_classes)                  

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
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=5):
        super(RegressionSegmentationModuleListConcatMasks, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.loss_type = loss_type

        self.sigmoid_heads = nn.ModuleList()
        self.s_heads = nn.ModuleList()

        for i in range(num_semantic_classes):       
            self.sigmoid_heads.append(SigmoidHead(128 + 1, 128, num_classes=1))
            self.s_heads.append(SSHead(128 + 1, num_classes=1))                

        self.relu = nn.ReLU(inplace=True)

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, masks):
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

class RegressionInstancesSemanticNoMaskingCanonical(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=14, feature_map_instances_dim=32, num_instances=63,var=0,padding_instances=0):
        super(RegressionInstancesSemanticNoMaskingCanonical, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.feature_map_instances_dim = feature_map_instances_dim
        self.num_instances = num_instances
        self.var = var
        self.project = ProjectionCustom(hidden_dim, 32, 128)
        self.padding_instances = padding_instances
        global padding_global
        padding_global = padding_instances
  
        self.instances_scale_and_shift = ROISelectScale(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
        self.instances_canonical = ROISelectCanonical(128, 4, num_semantic_classes=self.num_semantic_classes-1)       
        if var == 1:
            self.instances_scale_and_shift = ROISelectScaleA(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
            self.instances_canonical = ROISelectCanonical(128, 4, num_semantic_classes=self.num_semantic_classes-1)       
        if var == 2 or var == 14:
            self.instances_scale_and_shift = ROISelectScaleB(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
            self.instances_canonical = ROISelectCanonical(128, 4, num_semantic_classes=self.num_semantic_classes-1)       
        if var == 3:
            self.instances_scale_and_shift = ROISelectScale(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
            self.instances_canonical = ROISelectCanonicalA(128, 4, num_semantic_classes=self.num_semantic_classes-1)   
        if var == 4:
            self.instances_scale_and_shift = ROISelectScale(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
            self.instances_canonical = ROISelectCanonicalB(128, 4, num_semantic_classes=self.num_semantic_classes-1)   
        if var == 5:
            self.instances_scale_and_shift = ROISelectScale(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
            self.instances_canonical = ROISelectCanonicalC(128, 4, num_semantic_classes=self.num_semantic_classes-1)   
        if var == 6:
            self.instances_scale_and_shift = ROISelectScaleB(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
            self.instances_canonical = ROISelectCanonicalC(128, 4, num_semantic_classes=self.num_semantic_classes-1)          
        if var == 7 or var == 15:
            self.instances_scale_and_shift = ROISelectScaleA(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
            self.instances_canonical = ROISelectCanonicalC(128, 4, num_semantic_classes=self.num_semantic_classes-1) 
        if var == 10:
            self.instances_scale_and_shift = ROISelectScaleC(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
            self.instances_canonical = ROISelectCanonical(128, 4, num_semantic_classes=self.num_semantic_classes-1) 
        if var == 11:
            self.instances_scale_and_shift = ROISelectScale(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
            self.instances_canonical = ROISelectCanonicalD(128, 4, num_semantic_classes=self.num_semantic_classes-1) 
        if var == 12 or var == 13:
            self.instances_scale_and_shift = ROISelectScaleC(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
            self.instances_canonical = ROISelectCanonicalD(128, 4, num_semantic_classes=self.num_semantic_classes-1)

        #self.p_head = SigmoidHead(hidden_dim, hidden_dim, num_classes=self.num_semantic_classes) 
        #self.s_head = SSHead(hidden_dim, num_classes=self.num_semantic_classes)                 

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, masks, instances, boxes, labels):
        """
         depth:      is typically zeros #
         context:    feature map from early layers 
         input_feature_map: feature map from late layers  
        """
        pred_depths_r_list = []    # metric 
        pred_depths_instances_r_list = []
        pred_depths_rc_list = []   # canonical
        pred_depths_instances_rc_list = []

        pred_scale_list = []
        pred_shift_list = []
        pred_scale_instances_list = []
        pred_shift_instances_list = []
        b, _, h, w = depth.shape

        batch_size, i_dim, h, w = instances.shape

        input_feature_map_instances = input_feature_map
        hidd_size, h_hid, w_hid = input_feature_map_instances.shape[1:]


        # Change boxes #
        input_feature_map_instances_roi = roi_select_features(input_feature_map_instances, boxes, labels) 

        instances_scale_shift = self.instances_scale_and_shift(input_feature_map_instances_roi, boxes, labels)
        instances_canonical = self.instances_canonical(input_feature_map_instances_roi, boxes, labels)

        instances_scale, instances_shift = pick_predictions_instances_scale(instances_scale_shift, labels)
        pred_scale_instances_list.append(instances_scale)
        pred_shift_instances_list.append(instances_shift)

        instances_canonical = pick_predictions_instances_canonical(instances_canonical, labels)
        pred_depths_instances_rc_list.append(instances_canonical)
    
        #pred_prob = self.p_head(input_feature_map)        # b, 16*c, 88, 280
        #pred_scale = self.s_head(input_feature_map)       # b, 2*c
        # revert back 
        #pred_scale_list.append(pred_scale[:, ::2])  # b, c
        #pred_shift_list.append(pred_scale[:, 1::2]) # b, c
        
        # Canonical
        # b, 16*c, h, w - c*b, 16, h, w
        # b*c, 16, h, w
        #depth_rc = pred_prob
        #pred_depths_rc_list.append(depth_rc)
        
        # Metric
        if self.loss_type == 0:
            #depth_r = (self.relu(depth_rc * pred_scale[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale[:, 1::2].unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
            depth_instances_r = (self.relu(instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) + instances_shift.unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
        else:
            #depth_r = depth_rc * pred_scale[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale[:, 1::2].unsqueeze(-1).unsqueeze(-1)
            depth_instances_r = instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) + instances_shift.unsqueeze(-1).unsqueeze(-1)
            """
            if self.var == 8:
                depth_instances_r = instances_canonical * 50 * F.sigmoid(instances_scale.unsqueeze(-1).unsqueeze(-1)) + instances_shift.unsqueeze(-1).unsqueeze(-1)
            elif self.var == 9 or self.var == 13 or self.var == 14 or self.var == 15:
                depth_instances_r = instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) + instances_shift.unsqueeze(-1).unsqueeze(-1)
            else:
                depth_instances_r = instances_canonical * 20 * F.sigmoid(instances_scale.unsqueeze(-1).unsqueeze(-1)) + instances_shift.unsqueeze(-1).unsqueeze(-1)
                depth_instances_r = instances_canonical * 20 * F.sigmoid(instances_scale.unsqueeze(-1).unsqueeze(-1)) + instances_shift.unsqueeze(-1).unsqueeze(-1)
            """
       
        # depth_r: b, c, h, w
        #pred_depths_r_list.append(depth_r)
        pred_depths_instances_r_list.append(depth_instances_r)
        
        result = {}
        result["pred_depths_r_list"] = [None]#pred_depths_r_list
        result["pred_depths_rc_list"] = [None]#pred_depths_rc_list
        result["pred_scale_list"] = [None]#pred_scale_list
        result["pred_shift_list"] = [None]#pred_shift_list

        result["pred_depths_instances_r_list"] = pred_depths_instances_r_list
        result["pred_depths_instances_rc_list"] = pred_depths_instances_rc_list
        result["pred_scale_instances_list"] = pred_scale_instances_list
        result["pred_shift_instances_list"] = pred_shift_instances_list

        return result

class RegressionInstancesAgnostic(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=14, feature_map_instances_dim=32, num_instances=63,var=0, padding_instances=0):
        super(RegressionInstancesAgnostic, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.feature_map_instances_dim = feature_map_instances_dim
        self.num_instances = num_instances
        self.var = var
        self.project = ProjectionCustom(hidden_dim, 32, 128)
   
        self.padding_instances = padding_instances
        global padding_global
        padding_global = padding_instances
      
        self.instances_scale_and_shift = ROISelectScaleAgnostic(128, downsampling=4, num_semantic_classes=1)
        self.instances_canonical = ROISelectCanonicalAgnostic(128, 4, num_semantic_classes=1)       
        #self.p_head = SigmoidHead(hidden_dim, hidden_dim, num_classes=self.num_semantic_classes) 
        #self.s_head = SSHead(hidden_dim, num_classes=self.num_semantic_classes)                 

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, masks, instances, boxes, labels):
        """
         depth:      is typically zeros #
         context:    feature map from early layers 
         input_feature_map: feature map from late layers  
        """
        pred_depths_r_list = []    # metric 
        pred_depths_instances_r_list = []
        pred_depths_rc_list = []   # canonical
        pred_depths_instances_rc_list = []

        pred_scale_list = []
        pred_shift_list = []
        pred_scale_instances_list = []
        pred_shift_instances_list = []
        b, _, h, w = depth.shape

        batch_size, i_dim, h, w = instances.shape

        input_feature_map_instances = input_feature_map
        hidd_size, h_hid, w_hid = input_feature_map_instances.shape[1:]


        # Change boxes #
        input_feature_map_instances_roi = roi_select_features(input_feature_map_instances, boxes, labels) 

        instances_scale_shift = self.instances_scale_and_shift(input_feature_map_instances_roi, boxes, labels)

        instances_canonical_trim = self.instances_canonical(input_feature_map_instances_roi, boxes, labels)
        instances_scale_trim = instances_scale_shift[:, ::2]
        instances_shift_trim = instances_scale_shift[:, 1::2]
    
        valid_boxes = labels.view(batch_size * i_dim, 1)
        valid_boxes = torch.nonzero(valid_boxes != 0)
        instances_canonical = torch.zeros((batch_size*i_dim, 1, h, w)).to(instances_canonical_trim.device)
        instances_canonical[valid_boxes[:,0]] = instances_canonical_trim
        instances_canonical = instances_canonical.view(batch_size, i_dim, h, w)

        instances_scale = torch.zeros((batch_size*i_dim, 1)).to(instances_scale_trim.device)
        instances_scale[valid_boxes[:,0]] = instances_scale_trim
        instances_scale = instances_scale.view(batch_size, i_dim)

        instances_shift = torch.zeros((batch_size*i_dim, 1)).to(instances_shift_trim.device)
        instances_shift[valid_boxes[:,0]] = instances_shift_trim
        instances_shift = instances_shift.view(batch_size, i_dim)

        pred_scale_instances_list.append(instances_scale)
        pred_shift_instances_list.append(instances_shift)

        pred_depths_instances_rc_list.append(instances_canonical)
    
        #pred_prob = self.p_head(input_feature_map)        # b, 16*c, 88, 280
        #pred_scale = self.s_head(input_feature_map)       # b, 2*c
        # revert back 
        #pred_scale_list.append(pred_scale[:, ::2])  # b, c
        #pred_shift_list.append(pred_scale[:, 1::2]) # b, c
        
        # Canonical
        # b, 16*c, h, w - c*b, 16, h, w
        # b*c, 16, h, w
        #depth_rc = pred_prob
        #pred_depths_rc_list.append(depth_rc)
        
        # Metric
        if self.loss_type == 0:
            #depth_r = (self.relu(depth_rc * pred_scale[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale[:, 1::2].unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
            depth_instances_r = (self.relu(instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) + instances_shift.unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
        else:
            #depth_r = depth_rc * pred_scale[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale[:, 1::2].unsqueeze(-1).unsqueeze(-1)
            depth_instances_r = instances_canonical * 20 * F.sigmoid(instances_scale.unsqueeze(-1).unsqueeze(-1)) + instances_shift.unsqueeze(-1).unsqueeze(-1)
       
        # depth_r: b, c, h, w
        #pred_depths_r_list.append(depth_r)
        pred_depths_instances_r_list.append(depth_instances_r)
        
        result = {}
        result["pred_depths_r_list"] = [None]#pred_depths_r_list
        result["pred_depths_rc_list"] = [None]#pred_depths_rc_list
        result["pred_scale_list"] = [None]#pred_scale_list
        result["pred_shift_list"] = [None]#pred_shift_list

        result["pred_depths_instances_r_list"] = pred_depths_instances_r_list
        result["pred_depths_instances_rc_list"] = pred_depths_instances_rc_list
        result["pred_scale_instances_list"] = pred_scale_instances_list
        result["pred_shift_instances_list"] = pred_shift_instances_list

        return result

class RegressionInstancesSharedCanonicalBins(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=14, feature_map_instances_dim=32, num_instances=63,var=0, padding_instances=0):
        super(RegressionInstancesSharedCanonicalBins, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.feature_map_instances_dim = feature_map_instances_dim
        self.num_instances = num_instances
        self.var = var
        self.project = ProjectionCustom(hidden_dim, 32, 128)
        self.padding_instances = padding_instances
        global padding
        padding_global = padding_instances
      
        self.instances_scale_and_shift = ROISelectScale(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
        self.instances_canonical = ROISelectSharedCanonicalUniform(128, 4, num_semantic_classes=bin_num+1)       
        if var == 1:
            self.instances_canonical = ROISelectSharedCanonicalBigUniform(128, 4, num_semantic_classes=bin_num+1)          
        if var == 2:
            self.instances_canonical = ROISelectSharedCanonicalHugeUniform(128, 4, num_semantic_classes=bin_num+1)       
        if var == 3:
            self.instances_scale_and_shift = ROISelectScaleBig(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
        if var == 4:
            self.instances_scale_and_shift = ROISelectScaleHuge(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
        if var == 5:
            self.instances_scale_and_shift = ROISelectScaleSmall(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)

        #self.p_head = SigmoidHead(hidden_dim, hidden_dim, num_classes=self.num_semantic_classes) 
        #self.s_head = SSHead(hidden_dim, num_classes=self.num_semantic_classes)                 

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, masks, instances, boxes, labels):
        """
         depth:      is typically zeros #
         context:    feature map from early layers 
         input_feature_map: feature map from late layers  
        """
        pred_depths_r_list = []    # metric 
        pred_depths_instances_r_list = []
        pred_depths_rc_list = []   # canonical
        pred_depths_instances_rc_list = []

        pred_scale_list = []
        pred_shift_list = []
        pred_scale_instances_list = []
        pred_shift_instances_list = []

        uncertainty_maps_list = []
        pred_depths_c_list = [] 

        b, _, h, w = depth.shape

        batch_size, i_dim, h, w = instances.shape

        input_feature_map_instances = input_feature_map
        hidd_size, h_hid, w_hid = input_feature_map_instances.shape[1:]

      

        # Change boxes #
        input_feature_map_instances_roi = roi_select_features(input_feature_map_instances, boxes, labels) 
        instances_scale_shift = self.instances_scale_and_shift(input_feature_map_instances_roi, boxes, labels)
        instances_scale, instances_shift = pick_predictions_instances_scale(instances_scale_shift, labels)
        pred_scale_instances_list.append(instances_scale)
        pred_shift_instances_list.append(instances_shift)

        instances_canonical = self.instances_canonical(input_feature_map, boxes, labels)
        
        bins_map = get_uniform_bins(depth, min_depth, max_depth, bin_num)[0, :, :, :].unsqueeze(0)
        bins_map = torch.cat([bins_map] * instances_canonical.shape[0], dim=0)  

        depth_rc_current = (instances_canonical * bins_map.detach()).sum(1, keepdim=True)
        uncertainty_map_current = torch.sqrt((instances_canonical * ((bins_map.detach() - depth_rc_current.repeat(1, bin_num+1, 1, 1))**2)).sum(1, keepdim=True))
        uncertainty_map = uncertainty_map_current

        pred_label = get_label(torch.squeeze(depth_rc_current, 1), bins_map, bin_num).unsqueeze(1)
        depth_c  = (torch.gather(bins_map.detach(), 1, pred_label.detach()))
        
        pred_depths_c_list.append(depth_c)
        uncertainty_maps_list.append(uncertainty_map)
        instances_canonical = depth_rc_current
        
        valid_boxes = labels.view(batch_size * i_dim, 1)
        valid_boxes = torch.nonzero(valid_boxes != 0)
        size_boxes_sq = valid_boxes.shape[0]
        canonical_full = torch.zeros((batch_size*i_dim, 1, h, w)).to(instances_canonical.device)
        canonical_full[valid_boxes[:,0]] = instances_canonical 
        canonical_full = canonical_full.view(batch_size, i_dim, h, w)
        instances_canonical = canonical_full
        pred_depths_instances_rc_list.append(instances_canonical)
    
        #pred_prob = self.p_head(input_feature_map)        # b, 16*c, 88, 280
        #pred_scale = self.s_head(input_feature_map)       # b, 2*c
        # revert back 
        #pred_scale_list.append(pred_scale[:, ::2])  # b, c
        #pred_shift_list.append(pred_scale[:, 1::2]) # b, c
        
        # Canonical
        # b, 16*c, h, w - c*b, 16, h, w
        # b*c, 16, h, w
        #depth_rc = pred_prob
        #pred_depths_rc_list.append(depth_rc)
        
        # Metric
        if self.loss_type == 0:
            #depth_r = (self.relu(depth_rc * pred_scale[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale[:, 1::2].unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
            depth_instances_r = (self.relu(instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) + instances_shift.unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
        else:
            #depth_r = depth_rc * pred_scale[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale[:, 1::2].unsqueeze(-1).unsqueeze(-1)
            #if self.var == 8:
            #    depth_instances_r = instances_canonical * 50 * F.sigmoid(instances_scale.unsqueeze(-1).unsqueeze(-1)) + instances_shift.unsqueeze(-1).unsqueeze(-1)
            #elif self.var == 9 or self.var == 13 or self.var == 14 or self.var == 15:
            #    depth_instances_r = instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) + instances_shift.unsqueeze(-1).unsqueeze(-1)
            #else:
            depth_instances_r = instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) + instances_shift.unsqueeze(-1).unsqueeze(-1)
       
        # depth_r: b, c, h, w
        #pred_depths_r_list.append(depth_r)
        pred_depths_instances_r_list.append(depth_instances_r)
        
        result = {}
        result["pred_depths_r_list"] = [None]#pred_depths_r_list
        result["pred_depths_rc_list"] = [None]#pred_depths_rc_list
        result["pred_scale_list"] = [None]#pred_scale_list
        result["pred_shift_list"] = [None]#pred_shift_list
        result["pred_depths_c_list"] = pred_depths_c_list
        result["uncertainty_maps_list"] = uncertainty_maps_list

        result["pred_depths_instances_r_list"] = pred_depths_instances_r_list
        result["pred_depths_instances_rc_list"] = pred_depths_instances_rc_list
        result["pred_scale_instances_list"] = pred_scale_instances_list
        result["pred_shift_instances_list"] = pred_shift_instances_list

        return result

class RegressionInstancesSharedCanonical(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=14, feature_map_instances_dim=32, num_instances=63,var=0, padding_instances=0):
        super(RegressionInstancesSharedCanonical, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.feature_map_instances_dim = feature_map_instances_dim
        self.num_instances = num_instances
        self.var = var
        self.project = ProjectionCustom(hidden_dim, 32, 128)
        self.padding_instances = padding_instances
        global padding
        padding_global = padding_instances
      
        self.instances_scale_and_shift = ROISelectScale(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
        self.instances_canonical = ROISelectSharedCanonical(128, 4, num_semantic_classes=1)       
        if var == 1:
            self.instances_canonical = ROISelectSharedCanonicalBig(128, 4, num_semantic_classes=1)          
        if var == 2:
            self.instances_canonical = ROISelectSharedCanonicalHuge(128, 4, num_semantic_classes=1)       
        if var == 3:
            self.instances_scale_and_shift = ROISelectScaleBig(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
        if var == 4:
            self.instances_scale_and_shift = ROISelectScaleHuge(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
        if var == 5:
            self.instances_scale_and_shift = ROISelectScaleSmall(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)

        #self.p_head = SigmoidHead(hidden_dim, hidden_dim, num_classes=self.num_semantic_classes) 
        #self.s_head = SSHead(hidden_dim, num_classes=self.num_semantic_classes)                 

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, masks, instances, boxes, labels):
        """
         depth:      is typically zeros #
         context:    feature map from early layers 
         input_feature_map: feature map from late layers  
        """
        pred_depths_r_list = []    # metric 
        pred_depths_instances_r_list = []
        pred_depths_rc_list = []   # canonical
        pred_depths_instances_rc_list = []

        pred_scale_list = []
        pred_shift_list = []
        pred_scale_instances_list = []
        pred_shift_instances_list = []
        b, _, h, w = depth.shape

        batch_size, i_dim, h, w = instances.shape

        input_feature_map_instances = input_feature_map
        hidd_size, h_hid, w_hid = input_feature_map_instances.shape[1:]


        # Change boxes #
        input_feature_map_instances_roi = roi_select_features(input_feature_map_instances, boxes, labels) 
        instances_scale_shift = self.instances_scale_and_shift(input_feature_map_instances_roi, boxes, labels)
        instances_scale, instances_shift = pick_predictions_instances_scale(instances_scale_shift, labels)
        pred_scale_instances_list.append(instances_scale)
        pred_shift_instances_list.append(instances_shift)

        instances_canonical = self.instances_canonical(input_feature_map, boxes, labels)
        valid_boxes = labels.view(batch_size * i_dim, 1)
        valid_boxes = torch.nonzero(valid_boxes != 0)
        size_boxes_sq = valid_boxes.shape[0]
        canonical_full = torch.zeros((batch_size*i_dim, 1, h, w)).to(instances_canonical.device)
        canonical_full[valid_boxes[:,0]] = instances_canonical 
        canonical_full = canonical_full.view(batch_size, i_dim, h, w)
        instances_canonical = canonical_full
        pred_depths_instances_rc_list.append(instances_canonical)
    
        #pred_prob = self.p_head(input_feature_map)        # b, 16*c, 88, 280
        #pred_scale = self.s_head(input_feature_map)       # b, 2*c
        # revert back 
        #pred_scale_list.append(pred_scale[:, ::2])  # b, c
        #pred_shift_list.append(pred_scale[:, 1::2]) # b, c
        
        # Canonical
        # b, 16*c, h, w - c*b, 16, h, w
        # b*c, 16, h, w
        #depth_rc = pred_prob
        #pred_depths_rc_list.append(depth_rc)
        
        # Metric
        if self.loss_type == 0:
            #depth_r = (self.relu(depth_rc * pred_scale[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale[:, 1::2].unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
            depth_instances_r = (self.relu(instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) + instances_shift.unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
        else:
            #depth_r = depth_rc * pred_scale[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale[:, 1::2].unsqueeze(-1).unsqueeze(-1)
            if self.var == 8:
                depth_instances_r = instances_canonical * 50 * F.sigmoid(instances_scale.unsqueeze(-1).unsqueeze(-1)) + instances_shift.unsqueeze(-1).unsqueeze(-1)
            elif self.var == 9 or self.var == 13 or self.var == 14 or self.var == 15:
                depth_instances_r = instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) + instances_shift.unsqueeze(-1).unsqueeze(-1)
            else:
                depth_instances_r = instances_canonical * 20 * F.sigmoid(instances_scale.unsqueeze(-1).unsqueeze(-1)) + instances_shift.unsqueeze(-1).unsqueeze(-1)
       
        # depth_r: b, c, h, w
        #pred_depths_r_list.append(depth_r)
        pred_depths_instances_r_list.append(depth_instances_r)
        
        result = {}
        result["pred_depths_r_list"] = [None]#pred_depths_r_list
        result["pred_depths_rc_list"] = [None]#pred_depths_rc_list
        result["pred_scale_list"] = [None]#pred_scale_list
        result["pred_shift_list"] = [None]#pred_shift_list

        result["pred_depths_instances_r_list"] = pred_depths_instances_r_list
        result["pred_depths_instances_rc_list"] = pred_depths_instances_rc_list
        result["pred_scale_instances_list"] = pred_scale_instances_list
        result["pred_shift_instances_list"] = pred_shift_instances_list

        return result

class RegressionInstancesSharedCanonicalModule(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=14, feature_map_instances_dim=32, num_instances=63,var=0, padding_instances=0):
        super(RegressionInstancesSharedCanonicalModule, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.feature_map_instances_dim = feature_map_instances_dim
        self.num_instances = num_instances
        self.var = var
        self.project = ProjectionCustom(hidden_dim, 32, 128)
        self.padding_instances = padding_instances
        global padding
        padding_global = padding_instances
      
        self.instances_scale_and_shift = nn.ModuleList()
        for i in range(num_semantic_classes - 1):
            self.instances_scale_and_shift.append(ROISelectScaleModule(128, downsampling=4, num_semantic_classes=1))

        self.instances_canonical = ROISelectSharedCanonical(128, 4, num_semantic_classes=1)      

        if var == 1 or var == 3:
            for i in range(num_semantic_classes - 1):
                self.instances_scale_and_shift.append(ROISelectScaleModuleBig(128, downsampling=4, num_semantic_classes=1))        
        if var == 2 or var == 3:       
             self.instances_canonical = ROISelectSharedCanonicalBig(128, 4, num_semantic_classes=1)    
        
        #self.p_head = SigmoidHead(hidden_dim, hidden_dim, num_classes=self.num_semantic_classes) 
        #self.s_head = SSHead(hidden_dim, num_classes=self.num_semantic_classes)                 

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, masks, instances, boxes, labels):
        """
         depth:      is typically zeros #
         context:    feature map from early layers 
         input_feature_map: feature map from late layers  
        """
        pred_depths_r_list = []    # metric 
        pred_depths_instances_r_list = []
        pred_depths_rc_list = []   # canonical
        pred_depths_instances_rc_list = []

        pred_scale_list = []
        pred_shift_list = []
        pred_scale_instances_list = []
        pred_shift_instances_list = []
        b, _, h, w = depth.shape

        batch_size, i_dim, h, w = instances.shape

        input_feature_map_instances = input_feature_map
        hidd_size, h_hid, w_hid = input_feature_map_instances.shape[1:]

        valid_boxes_reshaped = labels.view(batch_size * i_dim, 1)
        valid_boxes = torch.nonzero(valid_boxes_reshaped != 0)

        # Change boxes #
        #input_feature_map_instances = self.project(input_feature_map_instances)

        instances_scale = torch.zeros((batch_size*i_dim, 1)).to(labels.device) 
        instances_shift = torch.zeros((batch_size*i_dim, 1)).to(labels.device)
 
        for i in range(self.num_semantic_classes-1):
            with torch.no_grad():
                valid_boxes_class = torch.nonzero(valid_boxes_reshaped == i + 1)

                if valid_boxes_class.shape[0] == 0:
                    continue

            input_feature_map_instances_roi = roi_select_features_module(input_feature_map_instances, boxes, labels, i+1) #[valid,hid,h,w]  
            scale, shift = self.instances_scale_and_shift[i](input_feature_map_instances_roi, boxes, labels, i+1)
            
            instances_scale[valid_boxes_class[:, 0]] = scale
            instances_shift[valid_boxes_class[:, 0]] = shift

        instances_scale = instances_scale.view(batch_size, i_dim)
        instances_shift = instances_shift.view(batch_size, i_dim)
        pred_scale_instances_list.append(instances_scale)
        pred_shift_instances_list.append(instances_shift)

        instances_canonical = self.instances_canonical(input_feature_map, boxes, labels)

        size_boxes_sq = valid_boxes.shape[0]
        canonical_full = torch.zeros((batch_size*i_dim, 1, h, w)).to(instances_canonical.device)
        canonical_full[valid_boxes[:,0]] = instances_canonical 
        canonical_full = canonical_full.view(batch_size, i_dim, h, w)
        instances_canonical = canonical_full
        pred_depths_instances_rc_list.append(instances_canonical)
    
        #pred_prob = self.p_head(input_feature_map)        # b, 16*c, 88, 280
        #pred_scale = self.s_head(input_feature_map)       # b, 2*c
        # revert back 
        #pred_scale_list.append(pred_scale[:, ::2])  # b, c
        #pred_shift_list.append(pred_scale[:, 1::2]) # b, c
        
        # Canonical
        # b, 16*c, h, w - c*b, 16, h, w
        # b*c, 16, h, w
        #depth_rc = pred_prob
        #pred_depths_rc_list.append(depth_rc)
        
        # Metric
        if self.loss_type == 0:
            #depth_r = (self.relu(depth_rc * pred_scale[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale[:, 1::2].unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
            depth_instances_r = (self.relu(instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) + instances_shift.unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
        else:
            #depth_r = depth_rc * pred_scale[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale[:, 1::2].unsqueeze(-1).unsqueeze(-1)
            depth_instances_r = instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) + instances_shift.unsqueeze(-1).unsqueeze(-1)
       
        # depth_r: b, c, h, w
        #pred_depths_r_list.append(depth_r)
        pred_depths_instances_r_list.append(depth_instances_r)
        
        result = {}
        result["pred_depths_r_list"] = [None]#pred_depths_r_list
        result["pred_depths_rc_list"] = [None]#pred_depths_rc_list
        result["pred_scale_list"] = [None]#pred_scale_list
        result["pred_shift_list"] = [None]#pred_shift_list

        result["pred_depths_instances_r_list"] = pred_depths_instances_r_list
        result["pred_depths_instances_rc_list"] = pred_depths_instances_rc_list
        result["pred_scale_instances_list"] = pred_scale_instances_list
        result["pred_shift_instances_list"] = pred_shift_instances_list

        return result

class RegressionInstancesNoSharedCanonicalModule(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=14, feature_map_instances_dim=32, num_instances=63,var=0, padding_instances=0):
        super(RegressionInstancesNoSharedCanonicalModule, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.feature_map_instances_dim = feature_map_instances_dim
        self.num_instances = num_instances
        self.var = var
        self.project = ProjectionCustom(hidden_dim, 32, 128)
        self.padding_instances = padding_instances
        global padding
        padding_global = padding_instances
      
        self.instances_scale_and_shift = nn.ModuleList()
        for i in range(num_semantic_classes - 1):
            self.instances_scale_and_shift.append(ROISelectScaleModule(128, downsampling=4, num_semantic_classes=1))

        self.instances_canonical = nn.ModuleList()
        for i in range(num_semantic_classes - 1):
            self.instances_canonical.append(ROISelectCanonicalModule(128, 4, num_semantic_classes=1))     
        
        if var == 1 or var == 3:
            for i in range(num_semantic_classes - 1):
                self.instances_scale_and_shift.append(ROISelectScaleModuleBig(128, downsampling=4, num_semantic_classes=1))        
        if var == 2 or var == 3:
            for i in range(num_semantic_classes - 1):
                self.instances_canonical.append(ROISelectCanonicalModuleBig(128, 4, num_semantic_classes=1))     

        #self.p_head = SigmoidHead(hidden_dim, hidden_dim, num_classes=self.num_semantic_classes) 
        #self.s_head = SSHead(hidden_dim, num_classes=self.num_semantic_classes)                 

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, masks, instances, boxes, labels):
        """
         depth:      is typically zeros #
         context:    feature map from early layers 
         input_feature_map: feature map from late layers  
        """
        pred_depths_r_list = []    # metric 
        pred_depths_instances_r_list = []
        pred_depths_rc_list = []   # canonical
        pred_depths_instances_rc_list = []

        pred_scale_list = []
        pred_shift_list = []
        pred_scale_instances_list = []
        pred_shift_instances_list = []
        b, _, h, w = depth.shape

        batch_size, i_dim, h, w = instances.shape

        input_feature_map_instances = input_feature_map
        hidd_size, h_hid, w_hid = input_feature_map_instances.shape[1:]

        valid_boxes_reshaped = labels.view(batch_size * i_dim, 1)
        valid_boxes = torch.nonzero(valid_boxes_reshaped != 0)

        # Change boxes #
        #input_feature_map_instances = self.project(input_feature_map_instances)

        instances_scale = torch.zeros((batch_size*i_dim, 1)).to(labels.device) 
        instances_shift = torch.zeros((batch_size*i_dim, 1)).to(labels.device)
        canonical_full = torch.zeros((batch_size*i_dim, 1, h, w)).to(labels.device)
 
        for i in range(self.num_semantic_classes-1):
            with torch.no_grad():
                valid_boxes_class = torch.nonzero(valid_boxes_reshaped == i + 1)

                if valid_boxes_class.shape[0] == 0:
                    continue

            input_feature_map_instances_roi = roi_select_features_module(input_feature_map_instances, boxes, labels, i+1) #[valid,hid,h,w]  
            scale, shift = self.instances_scale_and_shift[i](input_feature_map_instances_roi, boxes, labels, i+1)

            instances_canonical = self.instances_canonical[i](input_feature_map_instances_roi, boxes, labels, i+1)
            canonical_full[valid_boxes_class[:, 0]] = instances_canonical

            instances_scale[valid_boxes_class[:, 0]] = scale
            instances_shift[valid_boxes_class[:, 0]] = shift

        instances_canonical = canonical_full.view(batch_size, i_dim, h, w)
        instances_scale = instances_scale.view(batch_size, i_dim)
        instances_shift = instances_shift.view(batch_size, i_dim)
        pred_scale_instances_list.append(instances_scale)
        pred_shift_instances_list.append(instances_shift)
        pred_depths_instances_rc_list.append(instances_canonical)
    
        #pred_prob = self.p_head(input_feature_map)        # b, 16*c, 88, 280
        #pred_scale = self.s_head(input_feature_map)       # b, 2*c
        # revert back 
        #pred_scale_list.append(pred_scale[:, ::2])  # b, c
        #pred_shift_list.append(pred_scale[:, 1::2]) # b, c
        
        # Canonical
        # b, 16*c, h, w - c*b, 16, h, w
        # b*c, 16, h, w
        #depth_rc = pred_prob
        #pred_depths_rc_list.append(depth_rc)
        
        # Metric
        if self.loss_type == 0:
            #depth_r = (self.relu(depth_rc * pred_scale[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale[:, 1::2].unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
            depth_instances_r = (self.relu(instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) + instances_shift.unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
        else:
            #depth_r = depth_rc * pred_scale[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale[:, 1::2].unsqueeze(-1).unsqueeze(-1)
            depth_instances_r = instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) + instances_shift.unsqueeze(-1).unsqueeze(-1)

       
        # depth_r: b, c, h, w
        #pred_depths_r_list.append(depth_r)
        pred_depths_instances_r_list.append(depth_instances_r)
        
        result = {}
        result["pred_depths_r_list"] = [None]#pred_depths_r_list
        result["pred_depths_rc_list"] = [None]#pred_depths_rc_list
        result["pred_scale_list"] = [None]#pred_scale_list
        result["pred_shift_list"] = [None]#pred_shift_list

        result["pred_depths_instances_r_list"] = pred_depths_instances_r_list
        result["pred_depths_instances_rc_list"] = pred_depths_instances_rc_list
        result["pred_scale_instances_list"] = pred_scale_instances_list
        result["pred_shift_instances_list"] = pred_shift_instances_list

        return result


class RegressionInstancesPerClassC(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=14, feature_map_instances_dim=32, num_instances=63,var=0, padding_instances=0):
        super(RegressionInstancesPerClassC, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.feature_map_instances_dim = feature_map_instances_dim
        self.num_instances = num_instances
        self.var = var
        self.project = ProjectionCustom(hidden_dim, 32, 128)
        self.padding_instances = padding_instances
        padding_global = padding_instances
      
        self.instances_scale_and_shift = ROISelectScale(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
        self.instances_canonical = ROISelectCanonicalClass(128, 4, num_semantic_classes=self.num_semantic_classes-1)

        if var == 1 or var == 3:
            self.instances_scale_and_shift = ROISelectScaleBig(128, downsampling=4, num_semantic_classes=self.num_semantic_classes-1)
        if var == 2 or var == 3:
            self.instances_canonical = ROISelectCanonicalClassBig(128, 4, num_semantic_classes=self.num_semantic_classes-1)
        
        #self.p_head = SigmoidHead(hidden_dim, hidden_dim, num_classes=self.num_semantic_classes) 
        #self.s_head = SSHead(hidden_dim, num_classes=self.num_semantic_classes)                 

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, masks, instances, boxes, labels):
        """
         depth:      is typically zeros #
         context:    feature map from early layers 
         input_feature_map: feature map from late layers  
        """
        pred_depths_r_list = []    # metric 
        pred_depths_instances_r_list = []
        pred_depths_rc_list = []   # canonical
        pred_depths_instances_rc_list = []

        pred_scale_list = []
        pred_shift_list = []
        pred_scale_instances_list = []
        pred_shift_instances_list = []
        b, _, h, w = depth.shape

        batch_size, i_dim, h, w = instances.shape

        input_feature_map_instances = input_feature_map
        hidd_size, h_hid, w_hid = input_feature_map_instances.shape[1:]

        # Change boxes #
        input_feature_map_instances_roi = roi_select_features(input_feature_map_instances, boxes, labels)
        instances_scale_shift = self.instances_scale_and_shift(input_feature_map_instances_roi, boxes, labels)
        instances_scale, instances_shift = pick_predictions_instances_scale(instances_scale_shift, labels)

        input_feature_map_instances_roi_canonical = roi_select_features_canonical_shared(input_feature_map_instances, boxes, labels)
        input_feature_map_instances_roi_canonical = input_feature_map_instances_roi_canonical.view(batch_size*(self.num_semantic_classes-1), hidd_size, h_hid, w_hid)
        instances_canonical = self.instances_canonical(input_feature_map_instances_roi_canonical)
        
        instances_canonical = instances_canonical.view(batch_size*(self.num_semantic_classes-1)*(self.num_semantic_classes-1), h_hid, w_hid)
        instances_canonical = instances_canonical[torch.tensor(range(0, batch_size*(self.num_semantic_classes-1)*(self.num_semantic_classes-1), self.num_semantic_classes-1)).to(labels.device), :, :]
        instances_canonical = instances_canonical.view(batch_size, self.num_semantic_classes-1, h_hid, w_hid) 
        instances_canonical = instances_canonical.unsqueeze(1).repeat(1,i_dim, 1,1,1)
        instances_canonical = instances_canonical.view(batch_size*i_dim,self.num_semantic_classes-1, h_hid, w_hid)

        labels_fast = torch.where(labels == 0, 1, labels) 
        labels_fast -= 1
        labels_fast = labels_fast.view(batch_size*i_dim)
        instances_canonical = instances_canonical[torch.arange(batch_size*i_dim), labels_fast]
        instances_canonical = instances_canonical.view(batch_size, i_dim, h_hid, w_hid)
 
        pred_scale_instances_list.append(instances_scale)
        pred_shift_instances_list.append(instances_shift)

        pred_depths_instances_rc_list.append(instances_canonical)
    
        #pred_prob = self.p_head(input_feature_map)        # b, 16*c, 88, 280
        #pred_scale = self.s_head(input_feature_map)       # b, 2*c
        # revert back 
        #pred_scale_list.append(pred_scale[:, ::2])  # b, c
        #pred_shift_list.append(pred_scale[:, 1::2]) # b, c
        
        # Canonical
        # b, 16*c, h, w - c*b, 16, h, w
        # b*c, 16, h, w
        #depth_rc = pred_prob
        #pred_depths_rc_list.append(depth_rc)
        
        # Metric
        if self.loss_type == 0:
            #depth_r = (self.relu(depth_rc * pred_scale[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale[:, 1::2].unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
            depth_instances_r = (self.relu(instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) + instances_shift.unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
        else:
            #depth_r = depth_rc * pred_scale[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale[:, 1::2].unsqueeze(-1).unsqueeze(-1)
            if self.var == 8:
                depth_instances_r = instances_canonical * 50 * F.sigmoid(instances_scale.unsqueeze(-1).unsqueeze(-1)) + instances_shift.unsqueeze(-1).unsqueeze(-1)
            elif self.var == 9 or self.var == 13 or self.var == 14 or self.var == 15:
                depth_instances_r = instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) + instances_shift.unsqueeze(-1).unsqueeze(-1)
            else:
                depth_instances_r = instances_canonical * 20 * F.sigmoid(instances_scale.unsqueeze(-1).unsqueeze(-1)) + instances_shift.unsqueeze(-1).unsqueeze(-1)
       
        # depth_r: b, c, h, w
        #pred_depths_r_list.append(depth_r)
        pred_depths_instances_r_list.append(depth_instances_r)
        
        result = {}
        result["pred_depths_r_list"] = [None]#pred_depths_r_list
        result["pred_depths_rc_list"] = [None]#pred_depths_rc_list
        result["pred_scale_list"] = [None]#pred_scale_list
        result["pred_shift_list"] = [None]#pred_shift_list

        result["pred_depths_instances_r_list"] = pred_depths_instances_r_list
        result["pred_depths_instances_rc_list"] = pred_depths_instances_rc_list
        result["pred_scale_instances_list"] = pred_scale_instances_list
        result["pred_shift_instances_list"] = pred_shift_instances_list

        return result









class RegressionSemanticNoMaskingCanonicalConcProjMaskU(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=5, operation_mask='*'):
        super(RegressionSemanticNoMaskingCanonicalConcProjMaskU, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.operation_mask = operation_mask

        self.p_heads = nn.ModuleList()
        self.s_heads = nn.ModuleList()

        if operation_mask == None:
            extra_dim = 1
        else:
            extra_dim = 0

        for i in range(num_semantic_classes):       
            self.p_heads.append(SigmoidHead(128 + extra_dim, 128, num_classes=1))
            self.s_heads.append(SSHead(128 + extra_dim, num_classes=1))                

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, masks):
        """
         depth:      is typically zeros #
         context:    feature map from early layers 
         input_feature_map: feature map from late layers  
        """
        pred_depths_r_list = []    # metric 
        pred_depths_rc_list = []   # canonical
        pred_scale_list = []
        pred_shift_list = []

        b, _, h, w = depth.shape

        pred_prob = []
        pred_scale = [] 
        for i in range(self.num_semantic_classes):
            if self.operation_mask == None:
                input_feature_map_current = torch.cat((input_feature_map, masks[:, i, :, :].unsqueeze(1)), dim=1)
            if self.operation_mask == '+':
                input_feature_map_current = input_feature_map + masks[:, i, :, :].unsqueeze(1)
            if self.operation_mask == '*':
                input_feature_map_current = input_feature_map * masks[:, i, :, :].unsqueeze(1)

            prob = self.p_heads[i](input_feature_map_current)
            scale = self.s_heads[i](input_feature_map_current)

            pred_prob.append(prob)
            pred_scale.append(scale)

        pred_prob = torch.cat(pred_prob, dim=1) 
        pred_scale = torch.cat(pred_scale, dim=1) 

        # revert back 
        pred_scale_list.append(pred_scale[:, ::2])  # b, c
        pred_shift_list.append(pred_scale[:, 1::2]) # b, c
          
        # Canonical
        # b, 16*c, h, w - c*b, 16, h, w
        # b*c, 16, h, w
        depth_rc = pred_prob
        pred_depths_rc_list.append(depth_rc)
        
        # Metric
        if self.loss_type == 0:
            depth_r = (self.relu(depth_rc * pred_scale[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale[:, 1::2].unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
        else:
            depth_r = depth_rc * pred_scale[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale[:, 1::2].unsqueeze(-1).unsqueeze(-1)
        
        # depth_r: b, c, h, w
        pred_depths_r_list.append(depth_r)
        
        result = {}
        result["pred_depths_c_list"] = [] 
        result["uncertainty_maps_list"] = []
        result["pred_depths_r_list"] = pred_depths_r_list
        result["pred_depths_rc_list"] = pred_depths_rc_list
        result["pred_scale_list"] = pred_scale_list
        result["pred_shift_list"] = pred_shift_list

        return result




"""
Layers
"""

"""
PHead: propabilities bin prediction
"""        
class PHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, bin_num=16):
        super(PHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, bin_num, 3, padding=1)

    def forward(self, x):
        out = torch.softmax(self.conv2(F.relu(self.conv1(x))), 1)
        return out


"""
SigmoidHead: regression canonical prediction [0,1]
"""        
class SigmoidHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_classes=1):
        super(SigmoidHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, num_classes, 3, padding=1)

    def forward(self, x):
        out = torch.sigmoid(self.conv2(F.relu(self.conv1(x))))
        return out


"""
RegressionHead: regression prediction [-inf, inf]
"""        
class RegressionHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_classes=1):
        super(RegressionHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, num_classes, 3, padding=1)

    def forward(self, x):
        out = self.conv2(F.relu(self.conv1(x)))
        return out


"""
SSHead: scale and shift head - single per image 
"""     
class SSHead(nn.Module):
    def __init__(self, input_dim=128, num_classes=1):
        super(SSHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(88)
        self.fc1 = nn.Linear(88 * 88, num_classes * 2) 
    
    def forward(self, x):
        out = F.relu(self.pool(self.conv1(x)))
        out = torch.flatten(out, 1)
        out = self.fc1(out) 
       
        return out






"""
SHead: scale and shift prediction - per pixel
"""     
class SPHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super(SPHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)

    def forward(self, x):
        out = self.conv2(F.relu(self.conv1(x)))
        return out

"""
SSHead: scale and shift prediction - single per image 
"""     
class ScaleHead(nn.Module):
    def __init__(self, input_dim=128, num_classes=1):
        super(ScaleHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(88)
        self.fc1= nn.Linear(88 * 88, num_classes * 1) # Scale and shift 
    
    def forward(self, x):
        out = F.relu(self.pool(self.conv1(x)))
        out = torch.flatten(out, 1)
        #out = torch.sigmoid(self.fc1(out)) * 20
        out = F.relu(self.fc1(out)) 
        return out

"""
SSHead: scale and shift prediction - single per image 
"""     
class ShiftHead(nn.Module):
    def __init__(self, input_dim=128, num_classes=1):
        super(ShiftHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(88)
        self.fc1= nn.Linear(88 * 88, num_classes * 1) # Scale and shift 
    
    def forward(self, x):
        out = F.relu(self.pool(self.conv1(x)))
        out = torch.flatten(out, 1)
        #out = torch.sigmoid(self.fc1(out)) * 20
        out = self.fc1(out)  
        return out



"""
UHead: uncertainty prediction
"""  
class UPHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super(UPHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.conv2(F.relu(self.conv1(x))))
        return out

class ROISelectScale(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectScale, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1) # First preprocess ROI map 
        self.pool = nn.AdaptiveAvgPool2d(70) # 120
        self.fc1 = nn.Linear((70*70) + 4, 2*num_semantic_classes) # Scale and shift 
        
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

    def forward(self, x, boxes, labels):
        h, w = x.shape[2:]
        b, i, _ = boxes.shape

        boxes_tmp = boxes.view(b * i, 4)
        valid_boxes = labels.view(b * i, 1)
        valid_boxes = torch.nonzero(valid_boxes != 0)
        boxes_tmp = boxes_tmp[valid_boxes[:, 0]]

        i, _ = boxes_tmp.shape

        boxes_tmp = project_box_to_features(boxes_tmp, self.downsampling)
        normalized_box = normalize_box_v2(boxes_tmp, height=h, width=w)

        out = self.pool(F.relu(self.conv1(x)))

        out = torch.flatten(out, 1)
        out = torch.cat((out, normalized_box), dim=1)
        out = self.fc1(out)
        out = out.view(i, 2*self.num_semantic_classes)
        
        return out

class ROISelectScaleModule(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectScaleModule, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1) # First preprocess ROI map 
        self.pool = nn.AdaptiveAvgPool2d(70) # 120
        self.fc1 = nn.Linear((70*70) + 4, 2*num_semantic_classes) # Scale and shift 
        
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

    def forward(self, x, boxes, labels, class_label):

        with torch.no_grad():  # Are we sure for this? Yes, I think
            h, w = x.shape[2:]
            b, i, _ = boxes.shape
            boxes_tmp = boxes.view(b * i, 4)
            valid_boxes = labels.view(b * i, 1)
            valid_boxes = torch.nonzero(valid_boxes == class_label)
            boxes_tmp = boxes_tmp[valid_boxes[:, 0]]

            i, _ = boxes_tmp.shape

            boxes_tmp = project_box_to_features(boxes_tmp, self.downsampling)
            normalized_box = normalize_box_v2(boxes_tmp, height=h, width=w)

        out = self.pool(F.relu(self.conv1(x)))

        out = torch.flatten(out, 1)
        out = torch.cat((out, normalized_box), dim=1)
        out = self.fc1(out)
        out = out.view(i, 2*self.num_semantic_classes)

        scale = out[:, ::2]        
        shift = out[:, 1::2]

        return scale, shift

class ROISelectScaleModuleBig(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectScaleModuleBig, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, input_dim, 3, padding=1) # First preprocess ROI map 
        self.conv2 = nn.Conv2d(input_dim, 32, 3, padding=1) # First preprocess ROI map 
        self.conv3 = nn.Conv2d(32, 1, 3, padding=1) # First preprocess ROI map 
        self.pool = nn.AdaptiveAvgPool2d(70) # 120
        self.fc1 = nn.Linear((70*70) + 4, 2*num_semantic_classes) # Scale and shift 
        
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

    def forward(self, x, boxes, labels, class_label):

        with torch.no_grad():  # Are we sure for this? Yes, I think
            h, w = x.shape[2:]
            b, i, _ = boxes.shape
            boxes_tmp = boxes.view(b * i, 4)
            valid_boxes = labels.view(b * i, 1)
            valid_boxes = torch.nonzero(valid_boxes == class_label)
            boxes_tmp = boxes_tmp[valid_boxes[:, 0]]

            i, _ = boxes_tmp.shape

            boxes_tmp = project_box_to_features(boxes_tmp, self.downsampling)
            normalized_box = normalize_box_v2(boxes_tmp, height=h, width=w)

        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        out = self.pool(F.relu(self.conv3(out)))

        out = torch.flatten(out, 1)
        out = torch.cat((out, normalized_box), dim=1)
        out = self.fc1(out)
        out = out.view(i, 2*self.num_semantic_classes)

        scale = out[:, ::2]        
        shift = out[:, 1::2]

        return scale, shift

class ROISelectScaleSmall(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectScaleSmall, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1) # First preprocess ROI map 
        self.pool = nn.AdaptiveAvgPool2d(45) # 120
        self.fc1 = nn.Linear((45*45) + 4, 2*num_semantic_classes) # Scale and shift 
        
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

    def forward(self, x, boxes, labels):
        h, w = x.shape[2:]
        b, i, _ = boxes.shape

        boxes_tmp = boxes.view(b * i, 4)
        valid_boxes = labels.view(b * i, 1)
        valid_boxes = torch.nonzero(valid_boxes != 0)
        boxes_tmp = boxes_tmp[valid_boxes[:, 0]]

        i, _ = boxes_tmp.shape

        boxes_tmp = project_box_to_features(boxes_tmp, self.downsampling)
        normalized_box = normalize_box_v2(boxes_tmp, height=h, width=w)

        out = self.pool(F.relu(self.conv1(x)))

        out = torch.flatten(out, 1)
        out = torch.cat((out, normalized_box), dim=1)
        out = self.fc1(out)
        out = out.view(i, 2*self.num_semantic_classes)
        
        return out

class ROISelectScaleBig(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectScaleBig, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 96, 3, padding=1) # First preprocess ROI map 
        self.conv2 = nn.Conv2d(96, 1, 3, padding=1) # First preprocess ROI map 
        self.pool = nn.AdaptiveAvgPool2d(70) # 120
        self.fc1 = nn.Linear((70*70) + 4, 2*num_semantic_classes) # Scale and shift 
        
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

    def forward(self, x, boxes, labels):
        h, w = x.shape[2:]
        b, i, _ = boxes.shape

        boxes_tmp = boxes.view(b * i, 4)
        valid_boxes = labels.view(b * i, 1)
        valid_boxes = torch.nonzero(valid_boxes != 0)
        boxes_tmp = boxes_tmp[valid_boxes[:, 0]]

        i, _ = boxes_tmp.shape

        boxes_tmp = project_box_to_features(boxes_tmp, self.downsampling)
        normalized_box = normalize_box_v2(boxes_tmp, height=h, width=w)

        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))

        out = torch.flatten(out, 1)
        out = torch.cat((out, normalized_box), dim=1)
        out = self.fc1(out)
        out = out.view(i, 2*self.num_semantic_classes)
        
        return out

class ROISelectScaleHuge(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectScaleHuge, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, input_dim, 3, padding=1) # First preprocess ROI map 
        self.conv2 = nn.Conv2d(input_dim, 96, 3, padding=1) # First preprocess ROI map 
        self.conv3 = nn.Conv2d(96, 1, 3, padding=1) # First preprocess ROI map 
        self.pool = nn.AdaptiveAvgPool2d(70) # 120
        self.fc1 = nn.Linear((70*70) + 4, 800) # Scale and shift 
        self.fc2 = nn.Linear(800+4, 2*num_semantic_classes) # Scale and shift 
        
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

    def forward(self, x, boxes, labels):
        h, w = x.shape[2:]
        b, i, _ = boxes.shape

        boxes_tmp = boxes.view(b * i, 4)
        valid_boxes = labels.view(b * i, 1)
        valid_boxes = torch.nonzero(valid_boxes != 0)
        boxes_tmp = boxes_tmp[valid_boxes[:, 0]]

        i, _ = boxes_tmp.shape

        boxes_tmp = project_box_to_features(boxes_tmp, self.downsampling)
        normalized_box = normalize_box_v2(boxes_tmp, height=h, width=w)

        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        out = self.pool(F.relu(self.conv3(out)))

        out = torch.flatten(out, 1)
        out = torch.cat((out, normalized_box), dim=1)
        out = self.fc1(out)
        out = torch.cat((out, normalized_box), dim=1)
        out = self.fc2(out)
        out = out.view(i, 2*self.num_semantic_classes)
        
        return out

class ROISelectScaleAgnostic(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectScaleAgnostic, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 128, 3, padding=1) # First preprocess ROI map 
        self.conv2 = nn.Conv2d(128, 1, 3, padding=1) # First preprocess ROI map 
        self.pool = nn.AdaptiveAvgPool2d(70)
        self.fc1 = nn.Linear((70*70) + 4, 2*num_semantic_classes) # Scale and shift 
        
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

    def forward(self, x, boxes, labels):
        h, w = x.shape[2:]
        b, i, _ = boxes.shape

        boxes_tmp = boxes.view(b * i, 4)
        valid_boxes = labels.view(b * i, 1)
        valid_boxes = torch.nonzero(valid_boxes != 0)
        boxes_tmp = boxes_tmp[valid_boxes[:, 0]]

        i, _ = boxes_tmp.shape

        boxes_tmp = project_box_to_features(boxes_tmp, self.downsampling)
        normalized_box = normalize_box_v2(boxes_tmp, height=h, width=w)

        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))

        out = torch.flatten(out, 1)
        out = torch.cat((out, normalized_box), dim=1)
        out = self.fc1(out)
        out = out.view(i, 2*self.num_semantic_classes)
        
        return out

class ROISelectScaleA(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectScaleA, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1) # First preprocess ROI map 
        self.fc1 = nn.Linear((120*160) + 4, 2*num_semantic_classes) # Scale and shift 
        
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

    def forward(self, x, boxes, labels):
        h, w = x.shape[2:]
        b, i, _ = boxes.shape

        boxes_tmp = boxes.view(b * i, 4)
        valid_boxes = labels.view(b * i, 1)
        valid_boxes = torch.nonzero(valid_boxes != 0)
        boxes_tmp = boxes_tmp[valid_boxes[:, 0]]

        i, _ = boxes_tmp.shape

        boxes_tmp = project_box_to_features(boxes_tmp, self.downsampling)
        normalized_box = normalize_box_v2(boxes_tmp, height=h, width=w)

        out = F.relu(self.conv1(x))

        out = torch.flatten(out, 1)
        out = torch.cat((out, normalized_box), dim=1)
        out = self.fc1(out)
        out = out.view(i, 2*self.num_semantic_classes)
        
        return out

class ROISelectScaleB(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectScaleB, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 96, 3, padding=1) # First preprocess ROI map 
        self.conv2 = nn.Conv2d(96, 1, 3, padding=1) # First preprocess ROI map 
        self.fc1 = nn.Linear((120*160) + 4, 2*num_semantic_classes) # Scale and shift 
        
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

    def forward(self, x, boxes, labels):
        h, w = x.shape[2:]
        b, i, _ = boxes.shape

        boxes_tmp = boxes.view(b * i, 4)
        valid_boxes = labels.view(b * i, 1)
        valid_boxes = torch.nonzero(valid_boxes != 0)
        boxes_tmp = boxes_tmp[valid_boxes[:, 0]]

        i, _ = boxes_tmp.shape

        boxes_tmp = project_box_to_features(boxes_tmp, self.downsampling)
        normalized_box = normalize_box_v2(boxes_tmp, height=h, width=w)

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))

        out = torch.flatten(out, 1)
        out = torch.cat((out, normalized_box), dim=1)
        out = self.fc1(out)
        out = out.view(i, 2*self.num_semantic_classes)
        
        return out

class ROISelectScaleC(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectScaleC, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 96, 3, padding=1) # First preprocess ROI map 
        self.conv2 = nn.Conv2d(96, 1, 3, padding=1) # First preprocess ROI map 
        self.fc1 = nn.Linear((120*160) + 4, 500) # Scale and shift 
        self.fc2 = nn.Linear(500, 2*num_semantic_classes) # Scale and shift 
        
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

    def forward(self, x, boxes, labels):
        h, w = x.shape[2:]
        b, i, _ = boxes.shape

        boxes_tmp = boxes.view(b * i, 4)
        valid_boxes = labels.view(b * i, 1)
        valid_boxes = torch.nonzero(valid_boxes != 0)
        boxes_tmp = boxes_tmp[valid_boxes[:, 0]]

        i, _ = boxes_tmp.shape

        boxes_tmp = project_box_to_features(boxes_tmp, self.downsampling)
        normalized_box = normalize_box_v2(boxes_tmp, height=h, width=w)

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))

        out = torch.flatten(out, 1)
        out = torch.cat((out, normalized_box), dim=1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        out = out.view(i, 2*self.num_semantic_classes)
        
        return out

class CRIHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_classes=1):
        super(CRIHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, num_classes, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = torch.sigmoid(self.conv2(out))

        return out

class CRIHeadUniform(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_classes=1):
        super(CRIHeadUniform, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, num_classes, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = torch.softmax(self.conv2(out), 1)

        return out

class CRIHeadClassBig(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_classes=1):
        super(CRIHeadClassBig, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, num_classes, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(x))
        out = torch.sigmoid(self.conv3(out))

        return out

class CRIHeadSharedBig(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_classes=1):
        super(CRIHeadSharedBig, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, num_classes, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = torch.sigmoid(self.conv3(out))

        return out

class CRIHeadSharedBigUniform(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_classes=1):
        super(CRIHeadSharedBigUniform, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, num_classes, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = torch.softmax(self.conv3(out), 1)

        return out

class CRIHeadModuleBig(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_classes=1):
        super(CRIHeadModuleBig, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, num_classes, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = torch.sigmoid(self.conv3(out))

        return out

class CRIHeadBig(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_classes=1):
        super(CRIHeadBig, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, num_classes, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(x))
        out = torch.sigmoid(self.conv3(out))

        return out

class CRIHeadHuge(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_classes=1):
        super(CRIHeadHuge, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(x))
        out = F.relu(self.conv3(x))
        out = F.relu(self.conv4(x))
        out = torch.sigmoid(self.conv5(out))

        return out

class CRIHeadHugeUniform(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_classes=1):
        super(CRIHeadHugeUniform, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(x))
        out = F.relu(self.conv3(x))
        out = F.relu(self.conv4(x))
        out = torch.softmax(self.conv5(out), 1)

        return out

class CRIHeadA(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_classes=1):
        super(CRIHeadA, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, num_classes, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(x))
        out = torch.sigmoid(self.conv3(out))

        return out

class CRIHeadB(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_classes=1):
        super(CRIHeadB, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(input_dim, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, num_classes, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(x))
        out = torch.sigmoid(self.conv3(out))

        return out

class CRIHeadC(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_classes=1):
        super(CRIHeadC, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, 96, 3, padding=1)
        self.conv4 = nn.Conv2d(96, num_classes, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(x))
        out = F.relu(self.conv3(x))
        out = torch.sigmoid(self.conv4(out))

        return out


class ROISelectCanonical(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectCanonical, self).__init__()
              
        self.canonical_head = CRIHead(input_dim+4, hidden_dim=128, num_classes=num_semantic_classes) #128
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

    def forward(self, x, boxes, labels):

        h, w = x.shape[2:]
        b, i, _ = boxes.shape

        boxes_tmp = boxes.view(b * i, 4)
        valid_boxes = labels.view(b * i, 1)
        valid_boxes = torch.nonzero(valid_boxes != 0)
        boxes_tmp = boxes_tmp[valid_boxes[:, 0]]
        i, _ = boxes_tmp.shape

        boxes_tmp = project_box_to_features(boxes_tmp, self.downsampling)
        normalized_box = normalize_box_v2(boxes_tmp, height=h, width=w)
        normalized_box = normalized_box.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)
        out = torch.cat((x, normalized_box), dim=1)
        out = self.canonical_head(out)
        out = out.view(i, self.num_semantic_classes, h, w)
        
        return out

class ROISelectCanonicalModule(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectCanonicalModule, self).__init__()
              
        self.canonical_head = CRIHead(input_dim, hidden_dim=128, num_classes=num_semantic_classes) #128
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

    def forward(self, x, boxes, labels, class_label):
        with torch.no_grad():  # Are we sure for this? Yes, I think
            h, w = x.shape[2:]
            b, i, _ = boxes.shape
            boxes_tmp = boxes.view(b * i, 4)
            valid_boxes = labels.view(b * i, 1)
            valid_boxes = torch.nonzero(valid_boxes == class_label)
            boxes_tmp = boxes_tmp[valid_boxes[:, 0]]

            i, _ = boxes_tmp.shape

        out = self.canonical_head(x)
        out = out.view(i, self.num_semantic_classes, h, w)
        
        return out

class ROISelectCanonicalModuleBig(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectCanonicalModuleBig, self).__init__()
              
        self.canonical_head = CRIHeadModuleBig(input_dim, hidden_dim=128, num_classes=num_semantic_classes) #128
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

    def forward(self, x, boxes, labels, class_label):
        with torch.no_grad():  # Are we sure for this? Yes, I think
            h, w = x.shape[2:]
            b, i, _ = boxes.shape
            boxes_tmp = boxes.view(b * i, 4)
            valid_boxes = labels.view(b * i, 1)
            valid_boxes = torch.nonzero(valid_boxes == class_label)
            boxes_tmp = boxes_tmp[valid_boxes[:, 0]]

            i, _ = boxes_tmp.shape

        out = self.canonical_head(x)
        out = out.view(i, self.num_semantic_classes, h, w)
        
        return out

class ROISelectCanonicalClass(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectCanonicalClass, self).__init__()
              
        self.canonical_head = CRIHead(input_dim, hidden_dim=128, num_classes=num_semantic_classes)
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

    def forward(self, x):
        out = self.canonical_head(x)
        return out

class ROISelectCanonicalClassBig(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectCanonicalClassBig, self).__init__()
              
        self.canonical_head = CRIHeadClassBig(input_dim, hidden_dim=128, num_classes=num_semantic_classes)
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

    def forward(self, x):
        out = self.canonical_head(x)
        return out

class ROISelectSharedCanonical(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectSharedCanonical, self).__init__()
              
        self.canonical_head = CRIHead(input_dim, hidden_dim=128, num_classes=num_semantic_classes)
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

    def forward(self, x, boxes, labels):

        h, w = x.shape[2:]
        b, i, _ = boxes.shape

        boxes_tmp = boxes.view(b * i, 4)
        valid_boxes = labels.view(b * i, 1)
        valid_boxes = torch.nonzero(valid_boxes != 0)
        boxes_tmp = boxes_tmp[valid_boxes[:, 0]]
        i_dim, _ = boxes_tmp.shape

        instances_per_batch = torch.nonzero(labels != 0)
        instances_per_batch = torch.bincount(instances_per_batch[:, 0])

        out = self.canonical_head(x)
        out = torch.cat([out[i, :, :, :].unsqueeze(0).repeat(times,1,1,1) for i, times in enumerate(instances_per_batch)], dim=0)
        out = out.view(i_dim, 1, h, w)
        
        return out

class ROISelectSharedCanonicalUniform(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectSharedCanonicalUniform, self).__init__()
              
        self.canonical_head = CRIHeadUniform(input_dim, hidden_dim=128, num_classes=num_semantic_classes)
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim
        self.bin_num = 1 if self.num_semantic_classes <= 20 else self.num_semantic_classes

    def forward(self, x, boxes, labels):

        h, w = x.shape[2:]
        b, i, _ = boxes.shape

        boxes_tmp = boxes.view(b * i, 4)
        valid_boxes = labels.view(b * i, 1)
        valid_boxes = torch.nonzero(valid_boxes != 0)
        boxes_tmp = boxes_tmp[valid_boxes[:, 0]]
        i_dim, _ = boxes_tmp.shape

        instances_per_batch = torch.nonzero(labels != 0)
        instances_per_batch = torch.bincount(instances_per_batch[:, 0])

        out = self.canonical_head(x)
        out = torch.cat([out[i, :, :, :].unsqueeze(0).repeat(times,1,1,1) for i, times in enumerate(instances_per_batch)], dim=0)
        out = out.view(i_dim, self.bin_num, h, w)
        
        return out

class ROISelectSharedCanonicalBig(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectSharedCanonicalBig, self).__init__()
              
        self.canonical_head = CRIHeadSharedBig(input_dim, hidden_dim=128, num_classes=num_semantic_classes)
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

    def forward(self, x, boxes, labels):

        h, w = x.shape[2:]
        b, i, _ = boxes.shape

        boxes_tmp = boxes.view(b * i, 4)
        valid_boxes = labels.view(b * i, 1)
        valid_boxes = torch.nonzero(valid_boxes != 0)
        boxes_tmp = boxes_tmp[valid_boxes[:, 0]]
        i_dim, _ = boxes_tmp.shape

        instances_per_batch = torch.nonzero(labels != 0)
        instances_per_batch = torch.bincount(instances_per_batch[:, 0])

        out = self.canonical_head(x)
        out = torch.cat([out[i, :, :, :].unsqueeze(0).repeat(times,1,1,1) for i, times in enumerate(instances_per_batch)], dim=0)
        out = out.view(i_dim, 1, h, w)
        
        return out

class ROISelectSharedCanonicalBig(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectSharedCanonicalBig, self).__init__()
              
        self.canonical_head = CRIHeadBig(input_dim, hidden_dim=128, num_classes=num_semantic_classes)
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim
        self.bin_num = 1 if self.num_semantic_classes < 20 else num_semantic_classes

    def forward(self, x, boxes, labels):

        h, w = x.shape[2:]
        b, i, _ = boxes.shape

        boxes_tmp = boxes.view(b * i, 4)
        valid_boxes = labels.view(b * i, 1)
        valid_boxes = torch.nonzero(valid_boxes != 0)
        boxes_tmp = boxes_tmp[valid_boxes[:, 0]]
        i_dim, _ = boxes_tmp.shape

        instances_per_batch = torch.nonzero(labels != 0)
        instances_per_batch = torch.bincount(instances_per_batch[:, 0])

        out = self.canonical_head(x)
        out = torch.cat([out[i, :, :, :].unsqueeze(0).repeat(times,1,1,1) for i, times in enumerate(instances_per_batch)], dim=0)
        out = out.view(i_dim, self.bin_num, h, w)
        
        return out

class ROISelectSharedCanonicalBigUniform(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectSharedCanonicalBigUniform, self).__init__()
              
        self.canonical_head = CRIHeadSharedBigUniform(input_dim, hidden_dim=128, num_classes=num_semantic_classes)
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim
        self.bin_num = 1 if self.num_semantic_classes <= 20 else self.num_semantic_classes

    def forward(self, x, boxes, labels):

        h, w = x.shape[2:]
        b, i, _ = boxes.shape

        boxes_tmp = boxes.view(b * i, 4)
        valid_boxes = labels.view(b * i, 1)
        valid_boxes = torch.nonzero(valid_boxes != 0)
        boxes_tmp = boxes_tmp[valid_boxes[:, 0]]
        i_dim, _ = boxes_tmp.shape

        instances_per_batch = torch.nonzero(labels != 0)
        instances_per_batch = torch.bincount(instances_per_batch[:, 0])

        out = self.canonical_head(x)
        out = torch.cat([out[i, :, :, :].unsqueeze(0).repeat(times,1,1,1) for i, times in enumerate(instances_per_batch)], dim=0)
        out = out.view(i_dim, self.bin_num, h, w)
        
        return out

class ROISelectSharedCanonicalHuge(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectSharedCanonicalHuge, self).__init__()
              
        self.canonical_head = CRIHeadHuge(input_dim, hidden_dim=128, num_classes=num_semantic_classes)
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

    def forward(self, x, boxes, labels):

        h, w = x.shape[2:]
        b, i, _ = boxes.shape

        boxes_tmp = boxes.view(b * i, 4)
        valid_boxes = labels.view(b * i, 1)
        valid_boxes = torch.nonzero(valid_boxes != 0)
        boxes_tmp = boxes_tmp[valid_boxes[:, 0]]
        i_dim, _ = boxes_tmp.shape

        instances_per_batch = torch.nonzero(labels != 0)
        instances_per_batch = torch.bincount(instances_per_batch[:, 0])

        out = self.canonical_head(x)
        out = torch.cat([out[i, :, :, :].unsqueeze(0).repeat(times,1,1,1) for i, times in enumerate(instances_per_batch)], dim=0)
        out = out.view(i_dim, 1, h, w)
        
        return out

class ROISelectSharedCanonicalHugeUniform(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectSharedCanonicalHugeUniform, self).__init__()
              
        self.canonical_head = CRIHeadHugeUniform(input_dim, hidden_dim=128, num_classes=num_semantic_classes)
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim
        self.bin_num = 1 if self.num_semantic_classes <= 20 else self.num_semantic_classes

    def forward(self, x, boxes, labels):

        h, w = x.shape[2:]
        b, i, _ = boxes.shape

        boxes_tmp = boxes.view(b * i, 4)
        valid_boxes = labels.view(b * i, 1)
        valid_boxes = torch.nonzero(valid_boxes != 0)
        boxes_tmp = boxes_tmp[valid_boxes[:, 0]]
        i_dim, _ = boxes_tmp.shape

        instances_per_batch = torch.nonzero(labels != 0)
        instances_per_batch = torch.bincount(instances_per_batch[:, 0])

        out = self.canonical_head(x)
        out = torch.cat([out[i, :, :, :].unsqueeze(0).repeat(times,1,1,1) for i, times in enumerate(instances_per_batch)], dim=0)
        out = out.view(i_dim, self.bin_num, h, w)
        
        return out

class CRIHeadAg(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_classes=1):
        super(CRIHeadAg, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, num_classes, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = torch.sigmoid(self.conv2(out))

        return out

class ROISelectCanonicalAgnostic(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectCanonicalAgnostic, self).__init__()
              
        self.canonical_head = CRIHeadAg(input_dim, hidden_dim=128, num_classes=num_semantic_classes)
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

    def forward(self, x, boxes, labels):

        h, w = x.shape[2:]
        b, i, _ = boxes.shape

        boxes_tmp = boxes.view(b * i, 4)
        valid_boxes = labels.view(b * i, 1)
        valid_boxes = torch.nonzero(valid_boxes != 0)
        boxes_tmp = boxes_tmp[valid_boxes[:, 0]]
        i, _ = boxes_tmp.shape

        out = self.canonical_head(x)
        out = out.view(i, self.num_semantic_classes, h, w)
        
        return out

class ROISelectCanonicalA(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectCanonicalA, self).__init__()
              
        self.canonical_head = CRIHead(input_dim, hidden_dim=128, num_classes=num_semantic_classes)
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

    def forward(self, x, boxes, labels):

        h, w = x.shape[2:]
        b, i, _ = boxes.shape

        boxes_tmp = boxes.view(b * i, 4)
        valid_boxes = labels.view(b * i, 1)
        valid_boxes = torch.nonzero(valid_boxes != 0)
        boxes_tmp = boxes_tmp[valid_boxes[:, 0]]
        i, _ = boxes_tmp.shape

        out = self.canonical_head(x)
        out = out.view(i, self.num_semantic_classes, h, w)
        
        return out

class ROISelectCanonicalB(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectCanonicalB, self).__init__()
              
        self.canonical_head = CRIHeadA(input_dim, hidden_dim=128, num_classes=num_semantic_classes)
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

    def forward(self, x, boxes, labels):

        h, w = x.shape[2:]
        b, i, _ = boxes.shape

        boxes_tmp = boxes.view(b * i, 4)
        valid_boxes = labels.view(b * i, 1)
        valid_boxes = torch.nonzero(valid_boxes != 0)
        boxes_tmp = boxes_tmp[valid_boxes[:, 0]]
        i, _ = boxes_tmp.shape

        out = self.canonical_head(x)
        out = out.view(i, self.num_semantic_classes, h, w)
        
        return out

class ROISelectCanonicalC(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectCanonicalC, self).__init__()
              
        self.canonical_head = CRIHeadC(input_dim, hidden_dim=128, num_classes=num_semantic_classes)
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

    def forward(self, x, boxes, labels):

        h, w = x.shape[2:]
        b, i, _ = boxes.shape

        boxes_tmp = boxes.view(b * i, 4)
        valid_boxes = labels.view(b * i, 1)
        valid_boxes = torch.nonzero(valid_boxes != 0)
        boxes_tmp = boxes_tmp[valid_boxes[:, 0]]
        i, _ = boxes_tmp.shape

        out = self.canonical_head(x)
        out = out.view(i, self.num_semantic_classes, h, w)
        
        return out

class ROISelectCanonicalD(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectCanonicalD, self).__init__()
              
        self.canonical_head = CRIHeadB(input_dim, hidden_dim=128, num_classes=num_semantic_classes)
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

    def forward(self, x, boxes, labels):

        h, w = x.shape[2:]
        b, i, _ = boxes.shape

        boxes_tmp = boxes.view(b * i, 4)
        valid_boxes = labels.view(b * i, 1)
        valid_boxes = torch.nonzero(valid_boxes != 0)
        boxes_tmp = boxes_tmp[valid_boxes[:, 0]]
        i, _ = boxes_tmp.shape

        out = self.canonical_head(x)
        out = out.view(i, self.num_semantic_classes, h, w)
        
        return out


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128+192):
        super(SepConvGRU, self).__init__()

        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1))) 
        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h
