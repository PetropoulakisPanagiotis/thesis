import copy

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *

padding_global = 0

"""
IEBins metric depth implementation 
"""
class BasicUpdateBlockDepth(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16):
        super(BasicUpdateBlockDepth, self).__init__()

        self.encoder = ProjectionInputDepth(hidden_dim=hidden_dim, out_chs=hidden_dim * 2, bin_num=bin_num)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=self.encoder.out_chs+context_dim)
        self.p_head = PHead(hidden_dim, hidden_dim, bin_num)

    def forward(self, depth, context, gru_hidden, seq_len, bin_num, min_depth, max_depth):

        pred_depths_r_list = []
        pred_depths_c_list = []
        uncertainty_maps_list = []

        b, _, h, w = depth.shape
        depth_range = max_depth - min_depth
        interval = depth_range / bin_num
        interval = interval * torch.ones_like(depth)
        interval = interval.repeat(1, bin_num, 1, 1)
        interval = torch.cat([torch.ones_like(depth) * min_depth, interval], 1)

        bin_edges = torch.cumsum(interval, 1)
        current_depths = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        for i in range(seq_len):
            input_features = self.encoder(current_depths.detach())
            input_c = torch.cat([input_features, context], dim=1)

            gru_hidden = self.gru(gru_hidden, input_c)
            pred_prob = self.p_head(gru_hidden)

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

            bin_edges, current_depths = update_sample(bin_edges, target_bin_left, target_bin_right, depth_r.detach(), pred_label.detach(), bin_num, min_depth, max_depth, uncertainty_map)
        

        result = {}        
        result["pred_depths_r_list"] = pred_depths_r_list
        result["pred_depths_c_list"] = pred_depths_c_list
        result["uncertainty_maps_list"] = uncertainty_maps_list

        return result

"""
Canonical space basic block: one scale per semantic class and instance 
"""
class BasicUpdateBlockCSemanticMaskingDepth(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=5):
        super(BasicUpdateBlockCSemanticMaskingDepth, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=bin_num+context_dim)
        self.grus = []

        self.p_head = PHead(hidden_dim*self.num_semantic_classes, hidden_dim*self.num_semantic_classes, bin_num=bin_num * self.num_semantic_classes) # 16 propabilities canonical
        self.s_head = SSPHead(hidden_dim*self.num_semantic_classes, num_classes=self.num_semantic_classes)                 # Global scale and shift 

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, context, gru_hidden, seq_len, bin_num, min_depth, max_depth):
        """
         depth:      is typically zeros #
         context:    feature map from early layers 
         gru_hidden: feature map from late layers  
        """
        pred_depths_r_list = []    # metric 
        pred_depths_rc_list = []   # canonical
        pred_depths_c_list = []    # labels canonical
        uncertainty_maps_list = [] # std canonical 

        pred_scale_list = []
        pred_shift_list = []

        b, _, h, w = depth.shape
        
        depth_range = max_depth - min_depth
        interval = depth_range / bin_num
        interval = interval * torch.ones_like(depth)
        interval = interval.repeat(1, bin_num, 1, 1)
        interval = torch.cat([torch.ones_like(depth) * min_depth, interval], 1)

        bin_edges = torch.cumsum(interval, 1)
        current_depths = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
       
        bin_edges = torch.cat([bin_edges] * self.num_semantic_classes, dim=0)
        bin_edges = bin_edges.view(b * self.num_semantic_classes, bin_num + 1, h, w)   

        # current_depths c*b, 16, h, w
        current_depths = torch.cat([current_depths] * self.num_semantic_classes, dim=0)
        current_depths = current_depths.view(b * self.num_semantic_classes, bin_num, h, w) 

        context = torch.cat([context] * self.num_semantic_classes, dim=0)
        context = context.view(b * self.num_semantic_classes, self.context_dim, h, w)   

        gru_hidden = torch.cat([gru_hidden] * self.num_semantic_classes, dim=0)
        gru_hidden = gru_hidden.view(b * self.num_semantic_classes, self.hidden_dim, h, w)          

        for i in range(seq_len):
            input_c = torch.cat([current_depths.detach(), context], dim=1)  # c*b, h, h, w

            gru_hidden = self.gru(gru_hidden, input_c) # c*b, 128, 88, 280
            
            gru_hidden = gru_hidden.view(b, self.num_semantic_classes * self.hidden_dim, h,w) # b, c*128, 88, 280           
            pred_prob = self.p_head(gru_hidden)        # b, 16*c, 88, 280
            pred_scale = self.s_head(gru_hidden)       # b, 2*c
            
            # revert back 
            gru_hidden = gru_hidden.view(b*self.num_semantic_classes, self.hidden_dim, h,w) # b*c, 128, 88, 280           
            pred_scale_list.append(pred_scale[:, ::2])  # b, c
            pred_shift_list.append(pred_scale[:, 1::2]) # b, c
            
            # Canonical
            # b, 16*c, h, w - c*b, 16, h, w
            # b*c, 16, h, w
            pred_prob = pred_prob.view(b * self.num_semantic_classes, bin_num, h, w)
            depth_rc = (pred_prob * current_depths.detach()).sum(1, keepdim=True) # b*c, 1, h, w
            
            depth_rc = depth_rc.view(b, self.num_semantic_classes, h, w)
            pred_depths_rc_list.append(depth_rc)
            depth_rc = depth_rc.view(b * self.num_semantic_classes, 1, h, w)
       
            # Before: b*c, 1, h, w - b, c
            # After: b, c, h, w - b, c
            depth_rc = depth_rc.view(b, self.num_semantic_classes, h, w)
            # Metric
            if self.loss_type == 0:
                depth_r = (self.relu(depth_rc * pred_scale[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale[:, 1::2].unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
            else:
                depth_r = depth_rc * pred_scale[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale[:, 1::2].unsqueeze(-1).unsqueeze(-1)
            
            # depth_r: b, c, h, w
            pred_depths_r_list.append(depth_r)

            # std 
            # pred_prob: b*c, 16, h, w
            # current_depths: c*b, 16, h, w
            # depth_rc:  b, c, h, w  -> b*c, 1, h, w
            depth_rc = depth_rc.view(b * self.num_semantic_classes, 1, h, w)
            uncertainty_map = torch.sqrt((pred_prob * ((current_depths.detach() - depth_rc.repeat(1, bin_num, 1, 1))**2)).sum(1, keepdim=True))
            
            # b*c, 1, h, w    
            uncertainty_map = uncertainty_map.view(b, self.num_semantic_classes, h, w)
            uncertainty_maps_list.append(uncertainty_map)
            uncertainty_map = uncertainty_map.view(b * self.num_semantic_classes, 1, h, w)
            
            # label 
            # depth_rc:  b*c, 1, h, w
            # current_depths: c*b, 16, h, w
            pred_label = get_label(torch.squeeze(depth_rc, 1), bin_edges, bin_num).unsqueeze(1) # b*c, 1, h, w 
            depth_c = torch.gather(current_depths.detach(), 1, pred_label.detach())


            depth_c = depth_c.view(b, self.num_semantic_classes, h, w)
            pred_depths_c_list.append(depth_c)
            depth_c = depth_c.view(b * self.num_semantic_classes, 1, h, w)

            depth_rc = depth_rc.view(b*self.num_semantic_classes, 1, h, w)
            
            # select bin canditate
            label_target_bin_left = pred_label
            target_bin_left = torch.gather(bin_edges, 1, label_target_bin_left)
            label_target_bin_right = (pred_label.float() + 1).long()
            target_bin_right = torch.gather(bin_edges, 1, label_target_bin_right)

            # update edges and centers 
            # depth_rc: b*c, 1, h, w
            # bin_edges: b*c, 17, h, w
            # pred_label: b*c, 1, h, w
            # unc:      : b*c, 1, w
            bin_edges, current_depths = update_sample(bin_edges, target_bin_left, target_bin_right, depth_rc.detach(), pred_label.detach(), bin_num, min_depth, max_depth, uncertainty_map)
      
        result = {}
        result["pred_depths_r_list"] = pred_depths_r_list
        result["pred_depths_rc_list"] = pred_depths_rc_list
        result["pred_depths_c_list"] = pred_depths_c_list
        result["uncertainty_maps_list"] = uncertainty_maps_list
        result["pred_scale_list"] = pred_scale_list
        result["pred_shift_list"] = pred_shift_list

        return result

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

        #self.p_head = CRHead(hidden_dim, hidden_dim, num_classes=self.num_semantic_classes) 
        #self.s_head = SSPHead(hidden_dim, num_classes=self.num_semantic_classes)                 

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, context, gru_hidden, seq_len, bin_num, min_depth, max_depth, masks, instances, boxes, labels):
        """
         depth:      is typically zeros #
         context:    feature map from early layers 
         gru_hidden: feature map from late layers  
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

        gru_hidden_instances = gru_hidden
        hidd_size, h_hid, w_hid = gru_hidden_instances.shape[1:]


        # Change boxes #
        gru_hidden_instances_roi = roi_select_features(gru_hidden_instances, boxes, labels) 

        instances_scale_shift = self.instances_scale_and_shift(gru_hidden_instances_roi, boxes, labels)
        instances_canonical = self.instances_canonical(gru_hidden_instances_roi, boxes, labels)

        instances_scale, instances_shift = pick_predictions_instances_scale(instances_scale_shift, labels)
        pred_scale_instances_list.append(instances_scale)
        pred_shift_instances_list.append(instances_shift)

        instances_canonical = pick_predictions_instances_canonical(instances_canonical, labels)
        pred_depths_instances_rc_list.append(instances_canonical)
    
        #pred_prob = self.p_head(gru_hidden)        # b, 16*c, 88, 280
        #pred_scale = self.s_head(gru_hidden)       # b, 2*c
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
        #self.p_head = CRHead(hidden_dim, hidden_dim, num_classes=self.num_semantic_classes) 
        #self.s_head = SSPHead(hidden_dim, num_classes=self.num_semantic_classes)                 

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, context, gru_hidden, seq_len, bin_num, min_depth, max_depth, masks, instances, boxes, labels):
        """
         depth:      is typically zeros #
         context:    feature map from early layers 
         gru_hidden: feature map from late layers  
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

        gru_hidden_instances = gru_hidden
        hidd_size, h_hid, w_hid = gru_hidden_instances.shape[1:]


        # Change boxes #
        gru_hidden_instances_roi = roi_select_features(gru_hidden_instances, boxes, labels) 

        instances_scale_shift = self.instances_scale_and_shift(gru_hidden_instances_roi, boxes, labels)

        instances_canonical_trim = self.instances_canonical(gru_hidden_instances_roi, boxes, labels)
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
    
        #pred_prob = self.p_head(gru_hidden)        # b, 16*c, 88, 280
        #pred_scale = self.s_head(gru_hidden)       # b, 2*c
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

        #self.p_head = CRHead(hidden_dim, hidden_dim, num_classes=self.num_semantic_classes) 
        #self.s_head = SSPHead(hidden_dim, num_classes=self.num_semantic_classes)                 

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, context, gru_hidden, seq_len, bin_num, min_depth, max_depth, masks, instances, boxes, labels):
        """
         depth:      is typically zeros #
         context:    feature map from early layers 
         gru_hidden: feature map from late layers  
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

        gru_hidden_instances = gru_hidden
        hidd_size, h_hid, w_hid = gru_hidden_instances.shape[1:]


        # Change boxes #
        gru_hidden_instances_roi = roi_select_features(gru_hidden_instances, boxes, labels) 
        instances_scale_shift = self.instances_scale_and_shift(gru_hidden_instances_roi, boxes, labels)
        instances_scale, instances_shift = pick_predictions_instances_scale(instances_scale_shift, labels)
        pred_scale_instances_list.append(instances_scale)
        pred_shift_instances_list.append(instances_shift)

        instances_canonical = self.instances_canonical(gru_hidden, boxes, labels)
        valid_boxes = labels.view(batch_size * i_dim, 1)
        valid_boxes = torch.nonzero(valid_boxes != 0)
        size_boxes_sq = valid_boxes.shape[0]
        canonical_full = torch.zeros((batch_size*i_dim, 1, h, w)).to(instances_canonical.device)
        canonical_full[valid_boxes[:,0]] = instances_canonical 
        canonical_full = canonical_full.view(batch_size, i_dim, h, w)
        instances_canonical = canonical_full
        pred_depths_instances_rc_list.append(instances_canonical)
    
        #pred_prob = self.p_head(gru_hidden)        # b, 16*c, 88, 280
        #pred_scale = self.s_head(gru_hidden)       # b, 2*c
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
        
        #self.p_head = CRHead(hidden_dim, hidden_dim, num_classes=self.num_semantic_classes) 
        #self.s_head = SSPHead(hidden_dim, num_classes=self.num_semantic_classes)                 

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, context, gru_hidden, seq_len, bin_num, min_depth, max_depth, masks, instances, boxes, labels):
        """
         depth:      is typically zeros #
         context:    feature map from early layers 
         gru_hidden: feature map from late layers  
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

        gru_hidden_instances = gru_hidden
        hidd_size, h_hid, w_hid = gru_hidden_instances.shape[1:]

        valid_boxes_reshaped = labels.view(batch_size * i_dim, 1)
        valid_boxes = torch.nonzero(valid_boxes_reshaped != 0)

        # Change boxes #
        #gru_hidden_instances = self.project(gru_hidden_instances)

        instances_scale = torch.zeros((batch_size*i_dim, 1)).to(labels.device) 
        instances_shift = torch.zeros((batch_size*i_dim, 1)).to(labels.device)
 
        for i in range(self.num_semantic_classes-1):
            with torch.no_grad():
                valid_boxes_class = torch.nonzero(valid_boxes_reshaped == i + 1)

                if valid_boxes_class.shape[0] == 0:
                    continue

            gru_hidden_instances_roi = roi_select_features_module(gru_hidden_instances, boxes, labels, i+1) #[valid,hid,h,w]  
            scale, shift = self.instances_scale_and_shift[i](gru_hidden_instances_roi, boxes, labels, i+1)
            
            instances_scale[valid_boxes_class[:, 0]] = scale
            instances_shift[valid_boxes_class[:, 0]] = shift

        instances_scale = instances_scale.view(batch_size, i_dim)
        instances_shift = instances_shift.view(batch_size, i_dim)
        pred_scale_instances_list.append(instances_scale)
        pred_shift_instances_list.append(instances_shift)

        instances_canonical = self.instances_canonical(gru_hidden, boxes, labels)

        size_boxes_sq = valid_boxes.shape[0]
        canonical_full = torch.zeros((batch_size*i_dim, 1, h, w)).to(instances_canonical.device)
        canonical_full[valid_boxes[:,0]] = instances_canonical 
        canonical_full = canonical_full.view(batch_size, i_dim, h, w)
        instances_canonical = canonical_full
        pred_depths_instances_rc_list.append(instances_canonical)
    
        #pred_prob = self.p_head(gru_hidden)        # b, 16*c, 88, 280
        #pred_scale = self.s_head(gru_hidden)       # b, 2*c
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

        #self.p_head = CRHead(hidden_dim, hidden_dim, num_classes=self.num_semantic_classes) 
        #self.s_head = SSPHead(hidden_dim, num_classes=self.num_semantic_classes)                 

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, context, gru_hidden, seq_len, bin_num, min_depth, max_depth, masks, instances, boxes, labels):
        """
         depth:      is typically zeros #
         context:    feature map from early layers 
         gru_hidden: feature map from late layers  
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

        gru_hidden_instances = gru_hidden
        hidd_size, h_hid, w_hid = gru_hidden_instances.shape[1:]

        valid_boxes_reshaped = labels.view(batch_size * i_dim, 1)
        valid_boxes = torch.nonzero(valid_boxes_reshaped != 0)

        # Change boxes #
        #gru_hidden_instances = self.project(gru_hidden_instances)

        instances_scale = torch.zeros((batch_size*i_dim, 1)).to(labels.device) 
        instances_shift = torch.zeros((batch_size*i_dim, 1)).to(labels.device)
        canonical_full = torch.zeros((batch_size*i_dim, 1, h, w)).to(labels.device)
 
        for i in range(self.num_semantic_classes-1):
            with torch.no_grad():
                valid_boxes_class = torch.nonzero(valid_boxes_reshaped == i + 1)

                if valid_boxes_class.shape[0] == 0:
                    continue

            gru_hidden_instances_roi = roi_select_features_module(gru_hidden_instances, boxes, labels, i+1) #[valid,hid,h,w]  
            scale, shift = self.instances_scale_and_shift[i](gru_hidden_instances_roi, boxes, labels, i+1)

            instances_canonical = self.instances_canonical[i](gru_hidden_instances_roi, boxes, labels, i+1)
            canonical_full[valid_boxes_class[:, 0]] = instances_canonical

            instances_scale[valid_boxes_class[:, 0]] = scale
            instances_shift[valid_boxes_class[:, 0]] = shift

        instances_canonical = canonical_full.view(batch_size, i_dim, h, w)
        instances_scale = instances_scale.view(batch_size, i_dim)
        instances_shift = instances_shift.view(batch_size, i_dim)
        pred_scale_instances_list.append(instances_scale)
        pred_shift_instances_list.append(instances_shift)
        pred_depths_instances_rc_list.append(instances_canonical)
    
        #pred_prob = self.p_head(gru_hidden)        # b, 16*c, 88, 280
        #pred_scale = self.s_head(gru_hidden)       # b, 2*c
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
        
        #self.p_head = CRHead(hidden_dim, hidden_dim, num_classes=self.num_semantic_classes) 
        #self.s_head = SSPHead(hidden_dim, num_classes=self.num_semantic_classes)                 

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, context, gru_hidden, seq_len, bin_num, min_depth, max_depth, masks, instances, boxes, labels):
        """
         depth:      is typically zeros #
         context:    feature map from early layers 
         gru_hidden: feature map from late layers  
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

        gru_hidden_instances = gru_hidden
        hidd_size, h_hid, w_hid = gru_hidden_instances.shape[1:]

        # Change boxes #
        gru_hidden_instances_roi = roi_select_features(gru_hidden_instances, boxes, labels)
        instances_scale_shift = self.instances_scale_and_shift(gru_hidden_instances_roi, boxes, labels)
        instances_scale, instances_shift = pick_predictions_instances_scale(instances_scale_shift, labels)

        gru_hidden_instances_roi_canonical = roi_select_features_canonical_shared(gru_hidden_instances, boxes, labels)
        gru_hidden_instances_roi_canonical = gru_hidden_instances_roi_canonical.view(batch_size*(self.num_semantic_classes-1), hidd_size, h_hid, w_hid)
        instances_canonical = self.instances_canonical(gru_hidden_instances_roi_canonical)
        
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
    
        #pred_prob = self.p_head(gru_hidden)        # b, 16*c, 88, 280
        #pred_scale = self.s_head(gru_hidden)       # b, 2*c
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

"""
Canonical space basic block: one scale per semantic class and instance 
"""
class RegressionSemanticNoMaskingCanonicalConc(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=14):
        super(RegressionSemanticNoMaskingCanonicalConc, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        self.p_head = CRHead(hidden_dim + num_semantic_classes, hidden_dim, num_classes=self.num_semantic_classes) # 16 propabilities canonical
        self.s_head = SSPHead(hidden_dim + num_semantic_classes, num_classes=self.num_semantic_classes)                 # Global scale and shift 

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, context, gru_hidden, seq_len, bin_num, min_depth, max_depth, masks):
        """
         depth:      is typically zeros #
         context:    feature map from early layers 
         gru_hidden: feature map from late layers  
        """
        pred_depths_r_list = []    # metric 
        pred_depths_rc_list = []   # canonical

        pred_scale_list = []
        pred_shift_list = []

        b, _, h, w = depth.shape
        
        gru_hidden = torch.cat((gru_hidden, masks),dim=1)
        pred_prob = self.p_head(gru_hidden)        # b, 16*c, 88, 280
        pred_scale = self.s_head(gru_hidden)       # b, 2*c
       
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
        result["pred_depths_r_list"] = pred_depths_r_list
        result["pred_depths_rc_list"] = pred_depths_rc_list
        result["pred_scale_list"] = pred_scale_list
        result["pred_shift_list"] = pred_shift_list

        return result

class UniformSemanticNoMaskingCanonicalConc(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=14):
        super(UniformSemanticNoMaskingCanonicalConc, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.bin_num = bin_num
        self.p_head = CRHead(hidden_dim + num_semantic_classes, hidden_dim, num_classes=(bin_num + 1) * (self.num_semantic_classes)) # 16 propabilities canonical
        self.s_head = SSPHead(hidden_dim + num_semantic_classes, num_classes=self.num_semantic_classes)                 # Global scale and shift 

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, context, gru_hidden, seq_len, bin_num, min_depth, max_depth, masks):
        """
         depth:      is typically zeros #
         context:    feature map from early layers 
         gru_hidden: feature map from late layers  
        """
        pred_depths_r_list = []    # metric 
        pred_depths_rc_list = []   # canonical

        pred_scale_list = []
        pred_shift_list = []

        uncertainty_maps_list = []
        pred_depths_c_list = [] 

        b, _, h, w = depth.shape
        
        bins_map = get_uniform_bins(depth, min_depth, max_depth, bin_num)
        bins_map = torch.cat([bins_map] * self.num_semantic_classes, dim=0)        
        gru_hidden = torch.cat((gru_hidden, masks),dim=1)
        pred_prob = self.p_head(gru_hidden)        # b, 16*c, 88, 280
        pred_scale = self.s_head(gru_hidden)       # b, 2*c
       
        pred_prob = pred_prob.view(b*self.num_semantic_classes,bin_num+1,h,w)
        # revert back 
        pred_scale_list.append(pred_scale[:, ::2])  # b, c
        pred_shift_list.append(pred_scale[:, 1::2]) # b, c
          
        # Canonical
        # b, 16*c, h, w - c*b, 16, h, w
        # b*c, 16, h, w
        depth_rc = (pred_prob * bins_map.detach()).sum(1, keepdim=True)

        uncertainty_map = torch.sqrt((pred_prob * ((bins_map.detach() - depth_rc.repeat(1, bin_num+1, 1, 1))**2)).sum(1, keepdim=True))
        uncertainty_map = uncertainty_map.view(b, self.num_semantic_classes,h,w)
        uncertainty_maps_list.append(uncertainty_map)

        # Label #
        pred_label = get_label(torch.squeeze(depth_rc, 1), bins_map, bin_num).unsqueeze(1)
        depth_c = torch.gather(bins_map.detach(), 1, pred_label.detach())
        depth_c  = depth_c.view(b, self.num_semantic_classes, h, w)
        pred_depths_c_list.append(depth_c)
        
        depth_rc = depth_rc.view(b, self.num_semantic_classes, h, w)
        pred_depths_rc_list.append(depth_rc)
        
        # Metric
        if self.loss_type == 0:
            depth_r = (self.relu(depth_rc * pred_scale[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale[:, 1::2].unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
        else:
            depth_r = depth_rc * pred_scale[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale[:, 1::2].unsqueeze(-1).unsqueeze(-1)
        
        # depth_r: b, c, h, w
        pred_depths_r_list.append(depth_r)
        result = {}
        
        result["pred_depths_r_list"] = pred_depths_r_list
        result["pred_depths_rc_list"] = pred_depths_rc_list
        result["pred_scale_list"] = pred_scale_list
        result["pred_shift_list"] = pred_shift_list
        result["pred_depths_c_list"] = pred_depths_c_list
        result["uncertainty_maps_list"] = uncertainty_maps_list

        return result


"""
Canonical space basic block: one scale per semantic class and instance 
"""
class RegressionSemanticNoMaskingCanonical(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=5):
        super(RegressionSemanticNoMaskingCanonical, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        self.p_head = CRHead(hidden_dim, hidden_dim, num_classes=self.num_semantic_classes) # 16 propabilities canonical
        self.s_head = SSPHead(hidden_dim, num_classes=self.num_semantic_classes)                 # Global scale and shift 

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, context, gru_hidden, seq_len, bin_num, min_depth, max_depth, masks):
        """
         depth:      is typically zeros #
         context:    feature map from early layers 
         gru_hidden: feature map from late layers  
        """
        pred_depths_r_list = []    # metric 
        pred_depths_rc_list = []   # canonical

        pred_scale_list = []
        pred_shift_list = []

        b, _, h, w = depth.shape
        
        pred_prob = self.p_head(gru_hidden)        # b, 16*c, 88, 280
        pred_scale = self.s_head(gru_hidden)       # b, 2*c
       
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
Canonical space basic block: one scale per semantic class and instance 
"""
class RegressionSemanticNoMaskingCanonicalConc(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=5):
        super(RegressionSemanticNoMaskingCanonicalConc, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        self.p_head = CRHead(hidden_dim + num_semantic_classes, hidden_dim, num_classes=self.num_semantic_classes) # 16 propabilities canonical
        self.s_head = SSPHead(hidden_dim + num_semantic_classes, num_classes=self.num_semantic_classes)                 # Global scale and shift 

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, context, gru_hidden, seq_len, bin_num, min_depth, max_depth, masks):
        """
         depth:      is typically zeros #
         context:    feature map from early layers 
         gru_hidden: feature map from late layers  
        """
        pred_depths_r_list = []    # metric 
        pred_depths_rc_list = []   # canonical

        pred_scale_list = []
        pred_shift_list = []

        b, _, h, w = depth.shape
        
        gru_hidden = torch.cat((gru_hidden, masks),dim=1)
        pred_prob = self.p_head(gru_hidden)        # b, 16*c, 88, 280
        pred_scale = self.s_head(gru_hidden)       # b, 2*c
       
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
Canonical space basic block: one scale per semantic class and instance 
"""
class RegressionSemanticNoMaskingCanonicalConcProjMask(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=5, operation_mask='*'):
        super(RegressionSemanticNoMaskingCanonicalConcProjMask, self).__init__()
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
            self.p_heads.append(CRHead(128 + extra_dim, 128, num_classes=1))
            self.s_heads.append(SSPHead(128 + extra_dim, num_classes=1))                

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, context, gru_hidden, seq_len, bin_num, min_depth, max_depth, masks):
        """
         depth:      is typically zeros #
         context:    feature map from early layers 
         gru_hidden: feature map from late layers  
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
                gru_hidden_current = torch.cat((gru_hidden, masks[:, i, :, :].unsqueeze(1)), dim=1)
            if self.operation_mask == '+':
                gru_hidden_current = gru_hidden + masks[:, i, :, :].unsqueeze(1)
            if self.operation_mask == '*':
                gru_hidden_current = gru_hidden * masks[:, i, :, :].unsqueeze(1)

            prob = self.p_heads[i](gru_hidden_current)
            scale = self.s_heads[i](gru_hidden_current)

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

class RegressionSemanticNoMaskingCanonicalConcProjMaskBins(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=5, operation_mask='*'):
        super(RegressionSemanticNoMaskingCanonicalConcProjMaskBins, self).__init__()
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
            self.p_heads.append(CRHead(128 + extra_dim, 128, num_classes=bin_num+1))
            self.s_heads.append(SSPHead(128 + extra_dim, num_classes=1))                

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, context, gru_hidden, seq_len, bin_num, min_depth, max_depth, masks):
        """
         depth:      is typically zeros #
         context:    feature map from early layers 
         gru_hidden: feature map from late layers  
        """
        pred_depths_r_list = []    # metric 
        pred_depths_rc_list = []   # canonical
        pred_scale_list = []
        pred_shift_list = []
        uncertainty_maps_list = []
        pred_depths_c_list = [] 

        b, _, h, w = depth.shape
        depth_rc = []
        pred_scale = [] 
        bins_map = get_uniform_bins(depth, min_depth, max_depth, bin_num)

        uncertainty_map = torch.zeros(b, self.num_semantic_classes, h, w).to(masks.device)
        depth_c = torch.zeros(b, self.num_semantic_classes, h, w).to(masks.device)
        for i in range(self.num_semantic_classes):
            
            if self.operation_mask == None:
                gru_hidden_current = torch.cat((gru_hidden, masks[:, i, :, :].unsqueeze(1)), dim=1)
            if self.operation_mask == '+':
                gru_hidden_current = gru_hidden + masks[:, i, :, :].unsqueeze(1)
            if self.operation_mask == '*':
                gru_hidden_current = gru_hidden * masks[:, i, :, :].unsqueeze(1)

            prob = self.p_heads[i](gru_hidden_current)
            depth_rc_current = (prob * bins_map.detach()).sum(1, keepdim=True)

            uncertainty_map_current = torch.sqrt((prob * ((bins_map.detach() - depth_rc_current.repeat(1, bin_num+1, 1, 1))**2)).sum(1, keepdim=True))
            uncertainty_map[:, i, :, :] = uncertainty_map_current.squeeze(1)

            pred_label = get_label(torch.squeeze(depth_rc_current, 1), bins_map, bin_num).unsqueeze(1)
            depth_c[:, i, :, : ]  = (torch.gather(bins_map.detach(), 1, pred_label.detach())).squeeze(1)
            
            scale = self.s_heads[i](gru_hidden_current)

            depth_rc.append(depth_rc_current)
            pred_scale.append(scale)

        pred_depths_c_list.append(depth_c)
        uncertainty_maps_list.append(uncertainty_map)
        depth_rc = torch.cat(depth_rc, dim=1) 
        pred_scale = torch.cat(pred_scale, dim=1) 

        # revert back 
        pred_scale_list.append(pred_scale[:, ::2])  # b, c
        pred_shift_list.append(pred_scale[:, 1::2]) # b, c
          
        # Canonical
        # b, 16*c, h, w - c*b, 16, h, w
        # b*c, 16, h, w
        pred_depths_rc_list.append(depth_rc)
        
        # Metric
        if self.loss_type == 0:
            depth_r = (self.relu(depth_rc * pred_scale[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale[:, 1::2].unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
        else:
            depth_r = depth_rc * pred_scale[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale[:, 1::2].unsqueeze(-1).unsqueeze(-1)
        
        # depth_r: b, c, h, w
        pred_depths_r_list.append(depth_r)
        
        result = {}
        result["pred_depths_c_list"] = pred_depths_c_list
        result["uncertainty_maps_list"] = uncertainty_maps_list
        result["pred_depths_r_list"] = pred_depths_r_list
        result["pred_depths_rc_list"] = pred_depths_rc_list
        result["pred_scale_list"] = pred_scale_list
        result["pred_shift_list"] = pred_shift_list

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
            self.p_heads.append(CRHead(128 + extra_dim, 128, num_classes=1))
            self.s_heads.append(SSPHead(128 + extra_dim, num_classes=1))                

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, context, gru_hidden, seq_len, bin_num, min_depth, max_depth, masks):
        """
         depth:      is typically zeros #
         context:    feature map from early layers 
         gru_hidden: feature map from late layers  
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
                gru_hidden_current = torch.cat((gru_hidden, masks[:, i, :, :].unsqueeze(1)), dim=1)
            if self.operation_mask == '+':
                gru_hidden_current = gru_hidden + masks[:, i, :, :].unsqueeze(1)
            if self.operation_mask == '*':
                gru_hidden_current = gru_hidden * masks[:, i, :, :].unsqueeze(1)

            prob = self.p_heads[i](gru_hidden_current)
            scale = self.s_heads[i](gru_hidden_current)

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
Canonical space basic block: single scale per image 
"""
class BasicUpdateBlockCSNoProjectDepth(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0):
        super(BasicUpdateBlockCSNoProjectDepth, self).__init__()

        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=context_dim+bin_num)

        self.p_head = PHead(hidden_dim, hidden_dim, bin_num=bin_num) # propabilities canonical
        self.s_head = SSPHead(hidden_dim)                            # Global scale and shift 

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, context, gru_hidden, seq_len, bin_num, min_depth, max_depth):
        """
         depth:      is typically zeros #
         context:    feature map from early layers 
         gru_hidden: feature map from late layers  
        """
        pred_depths_r_list = []    # metric 
        pred_depths_rc_list = []   # canonical
        pred_depths_c_list = []    # labels canonical
        uncertainty_maps_list = [] # std canonical 

        pred_scale_list = []
        pred_shift_list = []

        b, _, h, w = depth.shape
        depth_range = max_depth - min_depth
        interval = depth_range / bin_num
        interval = interval * torch.ones_like(depth)
        interval = interval.repeat(1, bin_num, 1, 1)
        interval = torch.cat([torch.ones_like(depth) * min_depth, interval], 1)

        bin_edges = torch.cumsum(interval, 1)
        current_depths = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        for i in range(seq_len):
            input_c = torch.cat([current_depths.detach(), context], dim=1)  # input_c        352, 88, 280
            
            gru_hidden = self.gru(gru_hidden, input_c) # 128, 88, 280
             
            pred_prob = self.p_head(gru_hidden)        # 16, 88, 280
            pred_scale = self.s_head(gru_hidden)       # 2
            pred_scale_list.append(pred_scale[:, 0:1])
            pred_shift_list.append(pred_scale[:, 1:2])
            
            # Canonical
            depth_rc = (pred_prob * current_depths.detach()).sum(1, keepdim=True) # 1, 88, 280 
            pred_depths_rc_list.append(depth_rc)

            # Metric
            if self.loss_type == 0:
                depth_r = (self.relu(depth_rc * pred_scale[:, 0:1].unsqueeze(1).unsqueeze(1) + pred_scale[:, 1:2].unsqueeze(1).unsqueeze(1))).clamp(min=1e-3)
            else:
                depth_r = depth_rc * pred_scale[:, 0:1].unsqueeze(1).unsqueeze(1) + pred_scale[:, 1:2].unsqueeze(1).unsqueeze(1)

            pred_depths_r_list.append(depth_r)

            # std 
            uncertainty_map = torch.sqrt((pred_prob * ((current_depths.detach() - depth_rc.repeat(1, bin_num, 1, 1))**2)).sum(1, keepdim=True))
            uncertainty_maps_list.append(uncertainty_map)

            # label 
            pred_label = get_label(torch.squeeze(depth_rc, 1), bin_edges, bin_num).unsqueeze(1)
            depth_c = torch.gather(current_depths.detach(), 1, pred_label.detach())
            pred_depths_c_list.append(depth_c)

            # select bin canditate
            label_target_bin_left = pred_label
            target_bin_left = torch.gather(bin_edges, 1, label_target_bin_left)
            label_target_bin_right = (pred_label.float() + 1).long()
            target_bin_right = torch.gather(bin_edges, 1, label_target_bin_right)

            # update edges and centers  
            bin_edges, current_depths = update_sample(bin_edges, target_bin_left, target_bin_right, depth_rc.detach(), pred_label.detach(), bin_num, min_depth, max_depth, uncertainty_map)
        
        result = {}
        result["pred_depths_r_list"] = pred_depths_r_list
        result["pred_depths_rc_list"] = pred_depths_rc_list
        result["pred_depths_c_list"] = pred_depths_c_list
        result["uncertainty_maps_list"] = uncertainty_maps_list
        result["pred_scale_list"] = pred_scale_list
        result["pred_shift_list"] = pred_shift_list

        return result

"""
Canonical space basic block: single scale per image 
"""
class BasicUpdateBlockCSNoProjectDepth(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0):
        super(BasicUpdateBlockCSNoProjectDepth, self).__init__()

        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=context_dim+bin_num)

        #self.fc1= nn.Linear(88 * 88, num_classes * 2) # Scale and shift 
        self.p_head = PHead(hidden_dim, hidden_dim, bin_num=bin_num) # propabilities canonical
        self.s_head = SSPHead(hidden_dim)                            # Global scale and shift 

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, context, gru_hidden, seq_len, bin_num, min_depth, max_depth):
        """
         depth:      is typically zeros #
         context:    feature map from early layers 
         gru_hidden: feature map from late layers  
        """
        pred_depths_r_list = []    # metric 
        pred_depths_rc_list = []   # canonical
        pred_depths_c_list = []    # labels canonical
        uncertainty_maps_list = [] # std canonical 

        pred_scale_list = []
        pred_shift_list = []

        b, _, h, w = depth.shape
        depth_range = max_depth - min_depth
        interval = depth_range / bin_num
        interval = interval * torch.ones_like(depth)
        interval = interval.repeat(1, bin_num, 1, 1)
        interval = torch.cat([torch.ones_like(depth) * min_depth, interval], 1)

        bin_edges = torch.cumsum(interval, 1)
        current_depths = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        for i in range(seq_len):
            input_c = torch.cat([current_depths.detach(), context], dim=1)  # input_c        352, 88, 280
            
            gru_hidden = self.gru(gru_hidden, input_c) # 128, 88, 280
             
            pred_prob = self.p_head(gru_hidden)        # 16, 88, 280
            pred_scale = self.s_head(gru_hidden)       # 2
            pred_scale_list.append(pred_scale[:, 0:1])
            pred_shift_list.append(pred_scale[:, 1:2])
            
            # Canonical
            depth_rc = (pred_prob * current_depths.detach()).sum(1, keepdim=True) # 1, 88, 280 
            pred_depths_rc_list.append(depth_rc)

            # Metric
            if self.loss_type == 0:
                depth_r = (self.relu(depth_rc * pred_scale[:, 0:1].unsqueeze(1).unsqueeze(1) + pred_scale[:, 1:2].unsqueeze(1).unsqueeze(1))).clamp(min=1e-3)
            else:
                depth_r = depth_rc * pred_scale[:, 0:1].unsqueeze(1).unsqueeze(1) + pred_scale[:, 1:2].unsqueeze(1).unsqueeze(1)

            pred_depths_r_list.append(depth_r)

            # std 
            uncertainty_map = torch.sqrt((pred_prob * ((current_depths.detach() - depth_rc.repeat(1, bin_num, 1, 1))**2)).sum(1, keepdim=True))
            uncertainty_maps_list.append(uncertainty_map)

            # label 
            pred_label = get_label(torch.squeeze(depth_rc, 1), bin_edges, bin_num).unsqueeze(1)
            depth_c = torch.gather(current_depths.detach(), 1, pred_label.detach())
            pred_depths_c_list.append(depth_c)

            # select bin canditate
            label_target_bin_left = pred_label
            target_bin_left = torch.gather(bin_edges, 1, label_target_bin_left)
            label_target_bin_right = (pred_label.float() + 1).long()
            target_bin_right = torch.gather(bin_edges, 1, label_target_bin_right)

            # update edges and centers  
            bin_edges, current_depths = update_sample(bin_edges, target_bin_left, target_bin_right, depth_rc.detach(), pred_label.detach(), bin_num, min_depth, max_depth, uncertainty_map)
        
        result = {}
        result["pred_depths_r_list"] = pred_depths_r_list
        result["pred_depths_rc_list"] = pred_depths_rc_list
        result["pred_depths_c_list"] = pred_depths_c_list
        result["uncertainty_maps_list"] = uncertainty_maps_list
        result["pred_scale_list"] = pred_scale_list
        result["pred_shift_list"] = pred_shift_list

        return result

"""
Canonical space basic block: single scale per image 
"""
class UniformSingle(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0):
        super(UniformSingle, self).__init__()

        self.p_head = PHead(hidden_dim, hidden_dim, bin_num + 1) # propabilities canonical
        self.s_head = SSPHead(hidden_dim)                        # Global scale and shift 

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, context, gru_hidden, seq_len, bin_num, min_depth, max_depth):
        """
         depth:      is typically zeros #
         context:    feature map from early layers 
         gru_hidden: feature map from late layers  
        """
        pred_depths_r_list = []    # metric 
        pred_depths_rc_list = []   # canonical

        bins_map = get_uniform_bins(depth, min_depth, max_depth, bin_num)

        pred_scale_list = []
        pred_shift_list = []
        uncertainty_maps_list = []
        pred_depths_c_list = [] 
        
        pred_rc = self.p_head(gru_hidden)        # 16, 88, 280
        pred_scale = self.s_head(gru_hidden)       # 2
        pred_scale_list.append(pred_scale[:, 0:1])
        pred_shift_list.append(pred_scale[:, 1:2])
       
        # Canonical
        
        depth_rc = (pred_rc * bins_map.detach()).sum(1, keepdim=True)
        pred_depths_rc_list.append(depth_rc)

        uncertainty_map = torch.sqrt((pred_rc * ((bins_map.detach() - depth_rc.repeat(1, bin_num+1, 1, 1))**2)).sum(1, keepdim=True))
        uncertainty_maps_list.append(uncertainty_map)

        # Label #
        pred_label = get_label(torch.squeeze(depth_rc, 1), bins_map, bin_num).unsqueeze(1)
        depth_c = torch.gather(bins_map.detach(), 1, pred_label.detach())
        pred_depths_c_list.append(depth_c)

        # Metric
        if self.loss_type == 0:
            depth_r = (self.relu(depth_rc * pred_scale[:, 0:1].unsqueeze(1).unsqueeze(1) + pred_scale[:, 1:2].unsqueeze(1).unsqueeze(1))).clamp(min=1e-3)
        else:
            depth_r = depth_rc * pred_scale[:, 0:1].unsqueeze(1).unsqueeze(1) + pred_scale[:, 1:2].unsqueeze(1).unsqueeze(1)

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
Canonical space basic block: single scale per image 
"""
class Regression(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0):
        super(Regression, self).__init__()

        self.p_head = CRHead(hidden_dim, hidden_dim) # propabilities canonical
        self.s_head = SSPHead(hidden_dim)            # Global scale and shift 

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, context, gru_hidden, seq_len, bin_num, min_depth, max_depth):
        """
         depth:      is typically zeros #
         context:    feature map from early layers 
         gru_hidden: feature map from late layers  
        """
        pred_depths_r_list = []    # metric 
        pred_depths_rc_list = []   # canonical

        pred_scale_list = []
        pred_shift_list = []
        
         
        pred_rc = self.p_head(gru_hidden)        # 16, 88, 280
        pred_scale = self.s_head(gru_hidden)       # 2
        pred_scale_list.append(pred_scale[:, 0:1])
        pred_shift_list.append(pred_scale[:, 1:2])
        
        # Canonical
        depth_rc = pred_rc
        pred_depths_rc_list.append(depth_rc)

        # Metric
        if self.loss_type == 0:
            depth_r = (self.relu(depth_rc * pred_scale[:, 0:1].unsqueeze(1).unsqueeze(1) + pred_scale[:, 1:2].unsqueeze(1).unsqueeze(1))).clamp(min=1e-3)
        else:
            depth_r = depth_rc * pred_scale[:, 0:1].unsqueeze(1).unsqueeze(1) + pred_scale[:, 1:2].unsqueeze(1).unsqueeze(1)

        pred_depths_r_list.append(depth_r)

        result = {}
        result["pred_depths_r_list"] = pred_depths_r_list
        result["pred_depths_rc_list"] = pred_depths_rc_list
        result["pred_scale_list"] = pred_scale_list
        result["pred_shift_list"] = pred_shift_list

        return result

"""
PHead: propabilities bin prediction
"""        
class CRHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_classes=1):
        super(CRHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, num_classes, 3, padding=1)

    def forward(self, x):
        out = torch.sigmoid(self.conv2(F.relu(self.conv1(x))))
        return out

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
SSHead: scale and shift prediction - single per image 
"""     
class SSPHead(nn.Module):
    def __init__(self, input_dim=128, num_classes=1):
        super(SSPHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(88)
        self.fc1= nn.Linear(88 * 88, num_classes * 2) # Scale and shift 
    
    def forward(self, x):
        out = F.relu(self.pool(self.conv1(x)))
        out = torch.flatten(out, 1)
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

def pick_predictions_instances_scale(prediction, labels):

    batch_size, i_dim = labels.shape[0:2]

    labels_fast = torch.where(labels == 0, 1, labels) 
    labels_fast -= 1
    labels_fast = labels_fast.view(batch_size*i_dim)
    
    valid_boxes = labels.view(batch_size * i_dim, 1)
    valid_boxes = torch.nonzero(valid_boxes != 0)
    
    size_boxes_sq = valid_boxes.shape[0]

    labels_fast = labels_fast[valid_boxes[:,0]]
    
    pred_scale = prediction[:, ::2]
    pred_shift = prediction[:, 1::2]
   
    pred_scale = pred_scale[torch.arange(size_boxes_sq), labels_fast].unsqueeze(-1)
    pred_shift = pred_shift[torch.arange(size_boxes_sq), labels_fast].unsqueeze(-1)

    pred_scale_full = torch.zeros((batch_size*i_dim, 1)).to(prediction.device)
    pred_shift_full = torch.zeros((batch_size*i_dim, 1)).to(prediction.device)

    pred_scale_full[valid_boxes[:,0]] = pred_scale
    pred_shift_full[valid_boxes[:,0]] = pred_shift

    pred_scale_full = pred_scale_full.view(batch_size, i_dim)
    pred_shift_full = pred_shift_full.view(batch_size, i_dim)

    return pred_scale_full, pred_shift_full
    
def pick_predictions_instances_canonical(prediction, labels): 
    h, w = prediction.shape[2:]
    batch_size, i_dim = labels.shape[0:2]

    labels_fast = torch.where(labels == 0, 1, labels) 
    labels_fast -= 1
    labels_fast = labels_fast.view(batch_size*i_dim)
    
    valid_boxes = labels.view(batch_size * i_dim, 1)
    valid_boxes = torch.nonzero(valid_boxes != 0)
    
    size_boxes_sq = valid_boxes.shape[0]

    labels_fast = labels_fast[valid_boxes[:,0]]
    
    canonical = prediction[torch.arange(size_boxes_sq), labels_fast].unsqueeze(1)


    canonical_full = torch.zeros((batch_size*i_dim, 1, h, w)).to(prediction.device)
    canonical_full[valid_boxes[:,0]] = canonical 

    canonical_full = canonical_full.view(batch_size, i_dim, h, w)
    
    return canonical_full

def normalize_box(box, height=480, width=640):
    with torch.no_grad():
        return torch.stack(((box[:, :, 0] / height).float(), 
                            (box[:, :, 1] / width).float(), 
	                        (box[:, :, 2] / height).float(), 
	                        (box[:, :, 3] / width).float()), dim=2)

def normalize_box_v2(box, height=480, width=640):
    with torch.no_grad():
        return torch.stack(((box[:, 0] / height).float(), 
                            (box[:, 1] / width).float(), 
	                        (box[:, 2] / height).float(), 
	                        (box[:, 3] / width).float()), dim=1)

def project_box_to_features(box, downsampling, height=480, width=640, padding=0):
    padding = padding_global
    with torch.no_grad():
        if padding == 0:
            return torch.ceil(box / downsampling).int()
        else:
            new_box = torch.ceil(box / downsampling).int()
            height /= downsampling
            width /= downsampling
            new_box = torch.stack((
                (new_box[:, 0]-padding).clamp(min=0).int(), 
                (new_box[:, 1]-padding).clamp(min=0).int(), 
                (new_box[:, 2]+padding).clamp(max=height-1).int(), 
                (new_box[:, 3]+padding).clamp(max=width-1).int()), dim=1)
            return new_box 




def roi_select_features(feature_map, box, labels, downsampling=4):
    # Can be more efficient to skip zero maps?
    with torch.no_grad():
        batch_size, i_dim = box.shape[0:2]
        
        height, width = feature_map.size(-2), feature_map.size(-1)
        box_coordinates = box.view(batch_size * i_dim, 4)

        instances_per_batch = torch.nonzero(labels != 0)
        instances_per_batch = torch.bincount(instances_per_batch[:, 0])
        
        valid_boxes = labels.view(batch_size * i_dim, 1)
        valid_boxes = torch.nonzero(valid_boxes != 0)

        box_coordinates = box_coordinates[valid_boxes[:, 0]]
        i_dim = box_coordinates.shape[0] 

        box_coordinates = project_box_to_features(box_coordinates, downsampling)

        ymin, xmin, ymax, xmax = box_coordinates.split(1, dim=1)

        row_indices = torch.arange(height, device=feature_map.device).unsqueeze(0)
        col_indices = torch.arange(width, device=feature_map.device).unsqueeze(0)

        row_mask = (row_indices >= ymin) & (row_indices <= ymax)
        col_mask = (col_indices >= xmin) & (col_indices <= xmax)

        row_mask = row_mask.unsqueeze(1).unsqueeze(-1)
        col_mask = col_mask.unsqueeze(1).unsqueeze(2)
        zeros_mask = torch.cat([torch.zeros_like(feature_map[i, :, :, :].unsqueeze(0)).repeat(times,1,1,1) for i, times in enumerate(instances_per_batch)], dim=0)
        masks = (zeros_mask + row_mask) * (zeros_mask + col_mask)

        masked_feature_map = torch.cat([feature_map[i, :, :, :].unsqueeze(0).repeat(times,1,1,1) for i, times in enumerate(instances_per_batch)], dim=0) * masks
      
        if False: 
            for i in range(masked_feature_map.shape[0]): 
                x = upsample(masked_feature_map, 4)
                x = x[i, 0, :, :].unsqueeze(0).permute(1,2,0)
                x = (x - torch.min(x))/(torch.max(x) - torch.min(x))
                x = (x.cpu().detach().numpy() * 255).astype('uint8')
                cv2.imshow(str(i), x)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        return masked_feature_map

def roi_select_features_module(feature_map, box, labels, class_label, downsampling=4):
    # Can be more efficient to skip zero maps?
    with torch.no_grad():
        batch_size, i_dim = box.shape[0:2]
        
        height, width = feature_map.size(-2), feature_map.size(-1)
        box_coordinates = box.view(batch_size * i_dim, 4)

        instances_per_batch = torch.nonzero(labels == class_label)
        instances_per_batch = torch.bincount(instances_per_batch[:, 0])
        
        valid_boxes = labels.view(batch_size * i_dim, 1)
        valid_boxes = torch.nonzero(valid_boxes == class_label)

        box_coordinates = box_coordinates[valid_boxes[:, 0]]
        i_dim = box_coordinates.shape[0] 

        box_coordinates = project_box_to_features(box_coordinates, downsampling)

        ymin, xmin, ymax, xmax = box_coordinates.split(1, dim=1)

        row_indices = torch.arange(height, device=feature_map.device).unsqueeze(0)
        col_indices = torch.arange(width, device=feature_map.device).unsqueeze(0)

        row_mask = (row_indices >= ymin) & (row_indices <= ymax)
        col_mask = (col_indices >= xmin) & (col_indices <= xmax)

        row_mask = row_mask.unsqueeze(1).unsqueeze(-1)
        col_mask = col_mask.unsqueeze(1).unsqueeze(2)
        zeros_mask = torch.cat([torch.zeros_like(feature_map[i, :, :, :].unsqueeze(0)).repeat(times,1,1,1) for i, times in enumerate(instances_per_batch)], dim=0)
        masks = (zeros_mask + row_mask) * (zeros_mask + col_mask)

        masked_feature_map = torch.cat([feature_map[i, :, :, :].unsqueeze(0).repeat(times,1,1,1) for i, times in enumerate(instances_per_batch)], dim=0) * masks
      
        if False: 
            for i in range(masked_feature_map.shape[0]): 
                x = upsample(masked_feature_map, 4)
                x = x[i, 0, :, :].unsqueeze(0).permute(1,2,0)
                x = (x - torch.min(x))/(torch.max(x) - torch.min(x))
                x = (x.cpu().detach().numpy() * 255).astype('uint8')
                cv2.imshow(str(i), x)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        return masked_feature_map

def roi_select_features_canonical_shared(feature_map, box, labels, downsampling=4):
    # Can be more efficient to skip zero maps?
    with torch.no_grad():
        
        batch_size, i_dim = box.shape[0:2]
        height, width = feature_map.size(-2), feature_map.size(-1)
        feature_maps_final = torch.zeros((batch_size, 13, feature_map.shape[1], height, width)).to(labels.device)
        
        for class_i in range(13):
            i_dim = box.shape[1]
            box_coordinates = box.view(batch_size * i_dim, 4)

            instances_per_batch = torch.nonzero(labels == class_i)
            instances_per_batch = torch.bincount(instances_per_batch[:, 0])

            valid_boxes = labels.view(batch_size * i_dim, 1)
            valid_boxes = torch.nonzero(valid_boxes == class_i)
            if valid_boxes.shape[0] == 0:
                masked_feature_map = torch.zeros_like(feature_map)
            else:        
                box_coordinates = box_coordinates[valid_boxes[:, 0]]
                i_dim = box_coordinates.shape[0] 

                box_coordinates = project_box_to_features(box_coordinates, downsampling)

                ymin, xmin, ymax, xmax = box_coordinates.split(1, dim=1)

                row_indices = torch.arange(height, device=feature_map.device).unsqueeze(0)
                col_indices = torch.arange(width, device=feature_map.device).unsqueeze(0)

                row_mask = (row_indices >= ymin) & (row_indices <= ymax)
                col_mask = (col_indices >= xmin) & (col_indices <= xmax)

                row_mask = row_mask.unsqueeze(1).unsqueeze(-1)
                col_mask = col_mask.unsqueeze(1).unsqueeze(2)

                zeros_mask = torch.cat([torch.zeros_like(feature_map[i, :, :, :].unsqueeze(0)).repeat(times,1,1,1) for i, times in enumerate(instances_per_batch)], dim=0)

                instances_per_batch = torch.cat((torch.tensor([1]).to(labels.device), instances_per_batch))
                instances_per_batch = torch.cumsum(instances_per_batch, dim=0)

                masks = (zeros_mask + row_mask) * (zeros_mask + col_mask)
                batches_masks = torch.cat([torch.sum(masks[instances_per_batch[i] - 1:instances_per_batch[i+1]], dim=0).unsqueeze(0) for i in range(instances_per_batch.shape[0] - 1)], dim=0)
                masked_feature_map = feature_map * batches_masks
                feature_maps_final[:, class_i, :, :] = masked_feature_map

            if False: 
                for i in range(masked_feature_map.shape[0]): 
                    x = upsample(masked_feature_map, 4)
                    x = x[i, 0, :, :].unsqueeze(0).permute(1,2,0)
                    x = (x - torch.min(x))/(torch.max(x) - torch.min(x))
                    x = (x.cpu().detach().numpy() * 255).astype('uint8')
                    cv2.imshow(str(i), x)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
    
        return feature_maps_final

def roi_select_features_ag(feature_map, box, labels, downsampling=4):
    # Can be more efficient to skip zero maps?
    with torch.no_grad():
        batch_size, i_dim = box.shape[0:2]
        
        height, width = feature_map.size(-2), feature_map.size(-1)
        box_coordinates = box.view(batch_size * i_dim, 4)

        instances_per_batch = torch.nonzero(labels != 0)
        instances_per_batch = torch.bincount(instances_per_batch[:, 0])
        
        valid_boxes = labels.view(batch_size * i_dim, 1)
        valid_boxes = torch.nonzero(valid_boxes != 0)

        box_coordinates = box_coordinates[valid_boxes[:, 0]]
        i_dim = box_coordinates.shape[0] 

        box_coordinates = project_box_to_features(box_coordinates, downsampling)

        ymin, xmin, ymax, xmax = box_coordinates.split(1, dim=1)

        row_indices = torch.arange(height, device=feature_map.device).unsqueeze(0)
        col_indices = torch.arange(width, device=feature_map.device).unsqueeze(0)

        row_mask = (row_indices >= ymin) & (row_indices <= ymax)
        col_mask = (col_indices >= xmin) & (col_indices <= xmax)

        row_mask = row_mask.unsqueeze(1).unsqueeze(-1)
        col_mask = col_mask.unsqueeze(1).unsqueeze(2)

        zeros_mask = torch.cat([torch.zeros_like(feature_map[i, :, :, :].unsqueeze(0)).repeat(times,1,1,1) for i, times in enumerate(instances_per_batch)], dim=0)
        masks = (zeros_mask + row_mask) * (zeros_mask + col_mask)

        masked_feature_map = torch.cat([torch.cat([feature_map[i, :, :, :].unsqueeze(0).repeat(times,1,1,1) * masks[i], feature_map[i, :, :, :].unsqueeze(0).repeat(times,1,1,1)], dim=1) for i, times in enumerate(instances_per_batch)], dim=0)

        #x = upsample(masked_feature_map, 4)
        #x = x[0, 0, :, :].unsqueeze(0).permute(1,2,0)
        #x = (x - torch.min(x))/(torch.max(x) - torch.min(x))
        #x = (x.cpu().detach().numpy() * 255).astype('uint8')
        #cv2.imshow("instances_mapped_ittmage", x)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return masked_feature_map

def roi_select_features_ag(feature_map, box, labels, downsampling=4):
    # Can be more efficient to skip zero maps?
    with torch.no_grad():
        batch_size, i_dim = box.shape[0:2]
        
        height, width = feature_map.size(-2), feature_map.size(-1)
        box_coordinates = box.view(batch_size * i_dim, 4)

        instances_per_batch = torch.nonzero(labels != 0)
        instances_per_batch = torch.bincount(instances_per_batch[:, 0])
        
        valid_boxes = labels.view(batch_size * i_dim, 1)
        valid_boxes = torch.nonzero(valid_boxes != 0)

        box_coordinates = box_coordinates[valid_boxes[:, 0]]
        i_dim = box_coordinates.shape[0] 

        box_coordinates = project_box_to_features(box_coordinates, downsampling)

        ymin, xmin, ymax, xmax = box_coordinates.split(1, dim=1)

        row_indices = torch.arange(height, device=feature_map.device).unsqueeze(0)
        col_indices = torch.arange(width, device=feature_map.device).unsqueeze(0)

        row_mask = (row_indices >= ymin) & (row_indices <= ymax)
        col_mask = (col_indices >= xmin) & (col_indices <= xmax)

        row_mask = row_mask.unsqueeze(1).unsqueeze(-1)
        col_mask = col_mask.unsqueeze(1).unsqueeze(2)

        zeros_mask = torch.cat([torch.zeros_like(feature_map[i, :, :, :].unsqueeze(0)).repeat(times,1,1,1) for i, times in enumerate(instances_per_batch)], dim=0)
        masks = (zeros_mask + row_mask) * (zeros_mask + col_mask)

        masked_feature_map = torch.cat([torch.cat([feature_map[i, :, :, :].unsqueeze(0).repeat(times,1,1,1) * masks[i], feature_map[i, :, :, :].unsqueeze(0).repeat(times,1,1,1)], dim=1) for i, times in enumerate(instances_per_batch)], dim=0)

        #x = upsample(masked_feature_map, 4)
        #x = x[0, 0, :, :].unsqueeze(0).permute(1,2,0)
        #x = (x - torch.min(x))/(torch.max(x) - torch.min(x))
        #x = (x.cpu().detach().numpy() * 255).astype('uint8')
        #cv2.imshow("instances_mapped_ittmage", x)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return masked_feature_map

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

def update_sample(bin_edges, target_bin_left, target_bin_right, depth_r, pred_label, depth_num, min_depth, max_depth, uncertainty_range):
    """
    Update bins 
    """ 
    with torch.no_grad():    
        b, _, h, w = bin_edges.shape

        mode = 'direct'
        if mode == 'direct':
            depth_range = uncertainty_range
            depth_start_update = torch.clamp_min(depth_r - 0.5 * depth_range, min_depth)
        else:
            depth_range = uncertainty_range + (target_bin_right - target_bin_left).abs()
            depth_start_update = torch.clamp_min(target_bin_left - 0.5 * uncertainty_range, min_depth)

        interval = depth_range / depth_num
        interval = interval.repeat(1, depth_num, 1, 1)
        interval = torch.cat([torch.ones([b, 1, h, w], device=bin_edges.device) * depth_start_update, interval], 1)

        bin_edges = torch.cumsum(interval, 1).clamp(min_depth, max_depth)
        curr_depth = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        
    return bin_edges.detach(), curr_depth.detach()

def get_label(depth_prediction, bin_edges, depth_num):
    """
    Get bin label (index)
    """
    with torch.no_grad():
        label_bin = torch.zeros(depth_prediction.size(), dtype=torch.int64, device=depth_prediction.device)

        for i in range(depth_num):
            bin_mask = torch.ge(depth_prediction, bin_edges[:, i])
            bin_mask = torch.logical_and(bin_mask, torch.lt(depth_prediction, bin_edges[:, i + 1]))
        
            label_bin[bin_mask] = i
        
        return label_bin

def get_uniform_bins(map, min_depth=0, max_depth=0, bin_num=5):
    """
    Update bins 
    """ 
    with torch.no_grad():    
        b, _, h, w = map.shape

        interval = (max_depth - min_depth) / bin_num
        interval = torch.ones(b, bin_num + 1, h, w, device=map.device) * interval
        bins = torch.cumsum(interval, 1).clamp(min_depth, max_depth)

    return bins
