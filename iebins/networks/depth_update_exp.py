import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *

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
Canonical space basic block: one scale per pixel 
"""
class BasicUpdateBlockCDepth(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0):
        super(BasicUpdateBlockCDepth, self).__init__()

        self.encoder = ProjectionInputDepth(hidden_dim=hidden_dim, out_chs=hidden_dim * 2, bin_num=bin_num)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=self.encoder.out_chs+context_dim)

        self.p_head = PHead(hidden_dim, hidden_dim, bin_num=bin_num) # propabilities canonical
        self.s_head = SPHead(hidden_dim, hidden_dim)                 # scale and shift 

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
            input_features = self.encoder(current_depths.detach()) # current_depths 16, 88, 280
            input_c = torch.cat([input_features, context], dim=1)  # input_c        352, 88, 280
            
            gru_hidden = self.gru(gru_hidden, input_c) # 128, 88, 280
             
            pred_prob = self.p_head(gru_hidden)        # 16, 88, 280
            pred_scale = self.s_head(gru_hidden)       # 2, 88, 280
            pred_scale_list.append(pred_scale[:, 0:1, :, :])
            pred_shift_list.append(pred_scale[:, 1:2, :, :])
            
            # Canonical
            depth_rc = (pred_prob * current_depths.detach()).sum(1, keepdim=True) # 1, 88, 280 
            pred_depths_rc_list.append(depth_rc)
            
            # Metric
            if self.loss_type == 0:
                depth_r = (self.relu(depth_rc * pred_scale[:, 0:1, :, :] + pred_scale[:, 1:2, :, :])).clamp(min=1e-3)
            else:
                depth_r = depth_rc * pred_scale[:, 0:1, :, :] + pred_scale[:, 1:2, :, :]

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
class BasicUpdateBlockCSDepth(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0):
        super(BasicUpdateBlockCSDepth, self).__init__()

        self.encoder = ProjectionInputDepth(hidden_dim=hidden_dim, out_chs=hidden_dim * 2, bin_num=bin_num)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=self.encoder.out_chs+context_dim)

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
            input_features = self.encoder(current_depths.detach()) # current_depths 16, 88, 280
            input_c = torch.cat([input_features, context], dim=1)  # input_c        352, 88, 280
            
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
Canonical space basic block: one scale per semantic class and instance 
"""
class BasicUpdateBlockCSemanticDepth(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=14):
        super(BasicUpdateBlockCSemanticDepth, self).__init__()
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

"""
Canonical space basic block: one scale per semantic class and instance 
"""
class BasicUpdateBlockCSemanticMaskingDepth(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=14):
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

"""
Canonical space basic block: one scale per semantic class and instance 
"""
class RegressionSemanticMasking(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=14):
        super(RegressionSemanticMasking, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        self.p_head = CRHead(hidden_dim * self.num_semantic_classes, hidden_dim*self.num_semantic_classes, num_classes=self.num_semantic_classes) # 16 propabilities canonical
        self.s_head = SSPHead(hidden_dim*self.num_semantic_classes, num_classes=self.num_semantic_classes)                 # Global scale and shift 

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
        
        masks = masks.view(b * self.num_semantic_classes, 1, h, w)

        gru_hidden = torch.cat([gru_hidden] * self.num_semantic_classes, dim=1)
        gru_hidden = gru_hidden * masks
           
        gru_hidden = gru_hidden.view(b, self.num_semantic_classes * self.hidden_dim, h,w) # b*c, 128, 88, 280           
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

"""
Canonical space basic block: one scale per semantic class and instance 
"""
class RegressionSemanticNoMaskingSharedCanonical(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=14):
        super(RegressionSemanticNoMaskingSharedCanonical, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        self.p_head = CRHead(hidden_dim, hidden_dim, num_classes=1) # 16 propabilities canonical
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
        pred_prob = torch.cat([pred_prob] * self.num_semantic_classes, dim=1)
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

"""
Canonical space basic block: one scale per semantic class and instance 
"""
class RegressionSemanticNoMaskingCanonical(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=14):
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
        result["pred_depths_r_list"] = pred_depths_r_list
        result["pred_depths_rc_list"] = pred_depths_rc_list
        result["pred_scale_list"] = pred_scale_list
        result["pred_shift_list"] = pred_shift_list

        return result

class RegressionInstancesSemanticNoMaskingCanonical(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=14, feature_map_instances_dim=32, num_instances=63):
        super(RegressionInstancesSemanticNoMaskingCanonical, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.feature_map_instances_dim = feature_map_instances_dim
        self.num_instances = num_instances

        self.instances_scale_and_shift = ROISelectScale(hidden_dim, downsampling=4, num_semantic_classes=self.num_semantic_classes)
        self.instances_canonical = ROISelectCanonical(hidden_dim, hidden_dim, num_semantic_classes=self.num_semantic_classes)       # No void instance

        self.p_head = CRHead(hidden_dim, hidden_dim, num_classes=self.num_semantic_classes) 
        self.s_head = SSPHead(hidden_dim, num_classes=self.num_semantic_classes)                 

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
        hidd_size, h_hid, w_hid = gru_hidden.shape[1:]

        gru_hidden = torch.cat([gru_hidden] * i_dim, dim=1)
        gru_hidden = gru_hidden.view(batch_size * i_dim, hidd_size, h_hid, w_hid)
        boxes = boxes.view(batch_size * i_dim, 4)
        gru_hidden = roi_select_features(gru_hidden, boxes) 
        
        boxes = boxes.view(batch_size, i_dim, 4)
        instances_scale_shift = self.instances_scale_and_shift(gru_hidden, boxes)
        instances_canonical = self.instances_canonical(gru_hidden, boxes)

        instances_scale, instances_shift = pick_predictions_instances_scale(instances_scale_shift, labels)
        pred_scale_instances_list.append(instances_scale)
        pred_shift_instances_list.append(instances_shift)

        instances_canonical = pick_predictions_instances_canonical(instances_canonical, labels)
        pred_depths_instances_rc_list.append(instances_canonical)
    
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
            depth_instances_r = (self.relu(instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) + instances_shift.unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
        else:
            depth_r = depth_rc * pred_scale[:, ::2].unsqueeze(-1).unsqueeze(-1) + pred_scale[:, 1::2].unsqueeze(-1).unsqueeze(-1)
            depth_instances_r = instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) + instances_shift.unsqueeze(-1).unsqueeze(-1)
       
        # depth_r: b, c, h, w
        pred_depths_r_list.append(depth_r)
        pred_depths_instances_r_list.append(depth_instances_r)
        
        result = {}
        result["pred_depths_r_list"] = pred_depths_r_list
        result["pred_depths_rc_list"] = pred_depths_rc_list
        result["pred_scale_list"] = pred_scale_list
        result["pred_shift_list"] = pred_shift_list

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

"""
Canonical space basic block: one scale per semantic class and instance 
"""
class RegressionSemanticNoMaskingCanonicalConcProj(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=14):
        super(RegressionSemanticNoMaskingCanonicalConcProj, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.mask_dim = 32
        
        self.projection = ProjectionCustom(1, self.mask_dim, 96)

        
        self.p_head = CRHead(hidden_dim+self.mask_dim, hidden_dim, num_classes=1) # 16 propabilities canonical
        self.s_head = SSPHead(hidden_dim+self.mask_dim, num_classes=1)                 # Global scale and shift 

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

        gru_hidden = torch.cat([gru_hidden] * self.num_semantic_classes, dim=0)
        masks = masks.view(b * self.num_semantic_classes, 1, h, w)
        masks = self.projection(masks)

        gru_hidden = torch.cat((gru_hidden, masks), dim=1)
        pred_prob = self.p_head(gru_hidden)        # b, 16*c, 88, 280
        pred_scale = self.s_head(gru_hidden)       # b, 2*c
        pred_prob = pred_prob.view(b, self.num_semantic_classes, h, w)
        pred_scale = pred_scale.view(b, self.num_semantic_classes * 2)
 
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

"""
Canonical space basic block: one scale per semantic class and instance 
"""
class RegressionSemanticNoMaskingCanonicalConcProjMask(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=14, operation_mask='*'):
        super(RegressionSemanticNoMaskingCanonicalConcProjMask, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.operation_mask = operation_mask
        
        self.p_head = CRHead(hidden_dim, hidden_dim, num_classes=1) # 16 propabilities canonical
        self.s_head = SSPHead(hidden_dim, num_classes=1)                 # Global scale and shift 

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

        gru_hidden = torch.cat([gru_hidden] * self.num_semantic_classes, dim=0)
        masks = masks.view(b * self.num_semantic_classes, 1, h, w)
        
        if self.operation_mask == '*':
            gru_hidden = gru_hidden * masks
        else:
            gru_hidden = gru_hidden + masks

        pred_prob = self.p_head(gru_hidden)        # b, 16*c, 88, 280
        pred_scale = self.s_head(gru_hidden)       # b, 2*c
        pred_prob = pred_prob.view(b, self.num_semantic_classes, h, w)
        pred_scale = pred_scale.view(b, self.num_semantic_classes * 2)
 
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
Canonical space with uncertainty metric prediction
"""
class BasicUpdateBlockCUDepth(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0):
        super(BasicUpdateBlockCUDepth, self).__init__()

        self.encoder = ProjectionInputDepth(hidden_dim=hidden_dim, out_chs=hidden_dim * 2, bin_num=bin_num)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=self.encoder.out_chs+context_dim)

        self.p_head = PHead(hidden_dim, hidden_dim, bin_num=bin_num) # propabilities canonical
        self.s_head = SSPHead(hidden_dim)                            # Global scale and shift 
        self.u_head = UPHead(hidden_dim, hidden_dim)                 # uncertainty
 
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
        pred_depths_u_list = []    # uncertainty metric

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
            input_features = self.encoder(current_depths.detach()) # current_depths 16, 88, 280
            input_c = torch.cat([input_features, context], dim=1)  # input_c        352, 88, 280
            
            gru_hidden = self.gru(gru_hidden, input_c) # 128, 88, 280
             
            pred_prob = self.p_head(gru_hidden)        # 16, 88, 280
            pred_scale = self.s_head(gru_hidden)       # 2, 88, 280
            pred_scale_list.append(pred_scale[:, 0:1])
            pred_shift_list.append(pred_scale[:, 1:2])
            
            # Uncertainty 
            pred_u = self.u_head(gru_hidden)
            pred_depths_u_list.append(pred_u)

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
        result["pred_depths_u_list"] = pred_depths_u_list
        result["uncertainty_maps_list"] = uncertainty_maps_list
        result["pred_scale_list"] = pred_scale_list
        result["pred_shift_list"] = pred_shift_list

        return result

"""
Canonical space basic block with uncertainty concatenation
"""
class BasicUpdateBlockCUConcDepth(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0):
        super(BasicUpdateBlockCUConcDepth, self).__init__()

        self.encoder = ProjectionInputDepth(hidden_dim=hidden_dim, out_chs=hidden_dim * 2, bin_num=bin_num+1) # For uncertainty
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=self.encoder.out_chs+context_dim)

        self.p_head = PHead(hidden_dim, hidden_dim, bin_num=bin_num) # propabilities canonical
        self.s_head = SSPHead(hidden_dim)                            # Global scale and shift 

        self.relu = nn.ReLU(inplace=True)
        self.loss_type = loss_type

    def forward(self, depth, unc, context, gru_hidden, seq_len, bin_num, min_depth, max_depth):
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
            input_features = self.encoder(torch.cat([current_depths.detach(), unc.detach()], 1)) # current_depths 16+1, 88, 280
            input_c = torch.cat([input_features, context], dim=1)  # input_c        352, 88, 280
            
            gru_hidden = self.gru(gru_hidden, input_c) # 128, 88, 280
             
            pred_prob = self.p_head(gru_hidden)        # 16, 88, 280
            pred_scale = self.s_head(gru_hidden)       # 2, 88, 280
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
class RegressionConcact(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0):
        super(RegressionConcact, self).__init__()

        self.p_head = CRHead(hidden_dim+context_dim, hidden_dim) # propabilities canonical
        self.s_head = SSPHead(hidden_dim+context_dim)                            # Global scale and shift 

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
        
        input_c = torch.cat([gru_hidden, context], dim=1)  # input_c        352, 88, 280
        
         
        pred_rc = self.p_head(input_c)        # 16, 88, 280
        pred_scale = self.s_head(input_c)       # 2
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
Canonical space basic block: one scale per semantic class and instance 
"""
class BasicUpdateBlockCSemanticMaskingDepth(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=14):
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

    def forward(self, depth, context, gru_hidden, seq_len, bin_num, min_depth, max_depth, masks):
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
  
        masks = masks.view(b * self.num_semantic_classes, 1, h, w)

        # current_depths c*b, 16, h, w
        current_depths = torch.cat([current_depths] * self.num_semantic_classes, dim=0)
        current_depths = current_depths.view(b * self.num_semantic_classes, bin_num, h, w) 
        
        context = torch.cat([context] * self.num_semantic_classes, dim=0)
        context = context.view(b * self.num_semantic_classes, self.context_dim, h, w)   

        context = context * masks

        gru_hidden = torch.cat([gru_hidden] * self.num_semantic_classes, dim=0)
        gru_hidden = gru_hidden.view(b * self.num_semantic_classes, self.hidden_dim, h, w)          


        gru_hidden = gru_hidden * masks

        for i in range(seq_len):
            current_depths = current_depths * masks
            input_c = torch.cat([current_depths.detach(), context], dim=1)  # c*b, h, h, w

            gru_hidden = self.gru(gru_hidden, input_c) # c*b, 128, 88, 280
            gru_hidden = gru_hidden * masks 
   
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
        self.pool = nn.AdaptiveAvgPool2d(32)
        self.fc1= nn.Linear((32 * 32) + 4, 2*num_semantic_classes) # Scale and shift 
        
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

    def forward(self, x, boxes):
        normalized_box = normalize_box(boxes)
        h, w = x.shape[2:]
        
        b, i, _ = boxes.shape

        out = F.relu(self.pool(self.conv1(x)))
        out = torch.flatten(out, 1)

        normalized_box = normalized_box.view(b * i,  4)
        out = torch.cat((out, normalized_box), dim=1)
        out = self.fc1(out) 
        out = out.view(b*i, 2*self.num_semantic_classes)
        
        return out

class ROISelectCanonical(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectCanonical, self).__init__()
              
        self.canonical_head = CRHead(input_dim, hidden_dim=32, num_classes=num_semantic_classes)
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

    def forward(self, x, boxes):
        normalized_box = normalize_box(boxes)
        
        b, i, _ = boxes.shape
        h, w = x.shape[2:]
        
        out = self.canonical_head(x)
        out = out.view(b*i, self.num_semantic_classes, h, w)
        
        return out

def pick_predictions_instances_scale(predicion, labels):
    
    labels_fast = labels
    b, i = labels.shape[0:2]

    labels_fast = torch.where(labels_fast == -1, 0, labels_fast)
    
    labels_fast = labels_fast.view(b*i)

    pred_scale = predicion[:, ::2]
    pred_shift = predicion[:, 1::2]

    pred_scale = pred_scale[torch.arange(b*i), labels_fast]
    pred_shift = pred_shift[torch.arange(b*i), labels_fast]
    pred_scale = pred_scale.view(b, i)
    pred_shift = pred_shift.view(b, i)

    return pred_scale, pred_shift
    
def pick_predictions_instances_canonical(prediction, labels): 
    
    labels_fast = labels
    b, i = labels.shape[0:2]
    h, w = prediction.shape[2:]

    labels_fast = torch.where(labels_fast == -1, 0, labels_fast)
    
    labels_fast = labels_fast.view(b*i)

    canonical = prediction[torch.arange(b*i), labels_fast]
    canonical = canonical.unsqueeze(1)
    canonical = canonical.view(b, i, h, w)

    return canonical

def normalize_box(box, height=480, width=640):
    with torch.no_grad():
        return torch.stack(((box[:, :, 0] / height).float(), 
                            (box[:, :, 1] / width).float(), 
	                        (box[:, :, 2] / height).float(), 
	                        (box[:, :, 3] / width).float()), dim=2)

def project_box_to_features(box, downsampling):
    with torch.no_grad():
        return torch.ceil(box / downsampling).int()
   
def roi_select_features(feature_map, box_coordinates, downsampling=4):
    with torch.no_grad():
        batch_size = box_coordinates.size(0)
        height, width = feature_map.size(-2), feature_map.size(-1)

        ymin, xmin, ymax, xmax = box_coordinates.split(1, dim=1)

        row_indices = torch.arange(height, device=feature_map.device).unsqueeze(0)
        col_indices = torch.arange(width, device=feature_map.device).unsqueeze(0)

        row_mask = (row_indices >= ymin) & (row_indices < ymax)
        col_mask = (col_indices >= xmin) & (col_indices < xmax)

        row_mask = row_mask.unsqueeze(1).unsqueeze(-1) 
        col_mask = col_mask.unsqueeze(1).unsqueeze(2) 
        masks = (torch.zeros_like(feature_map) + row_mask) * (torch.zeros_like(feature_map) + col_mask)
        
        masked_feature_map = feature_map * masks

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
