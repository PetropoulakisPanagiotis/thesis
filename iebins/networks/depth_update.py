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
class BasicUpdateBlockCSIDepth(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=5):
        super(BasicUpdateBlockCSIDepth, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        self.encoder = ProjectionInputDepth(hidden_dim=hidden_dim, out_chs=hidden_dim * 2, bin_num=bin_num)

        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=self.encoder.out_chs+context_dim)
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
            input_features = self.encoder(current_depths.detach()) # c*b, x, h, w
            input_c = torch.cat([input_features, context], dim=1)  # c*b, h, h, w
            
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
SHead: scale and shift prediction
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
