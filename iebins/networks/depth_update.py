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
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, upsample_type=0):
        super(IEBINS, self).__init__()
        self.upsample_type = upsample_type

        self.encoder_project = ProjectionInputDepth(hidden_dim=hidden_dim, out_chs=hidden_dim * 2, bin_num=bin_num)

        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=self.encoder_project.out_chs + context_dim)

        self.p_head = PHead(hidden_dim, hidden_dim, bin_num)

    def forward(self, depth, context, gru_hidden, bin_num, min_depth, max_depth, max_tree_depth=6):

        pred_depths_r_list = []  # Metric
        pred_depths_c_list = []  # Labels
        uncertainty_maps_list = []

        # Create a feature map of size depth with the bin canditates values
        current_depths, bin_edges = get_iebins(depth, min_depth, max_depth, bin_num)

        for i in range(max_tree_depth):
            input_features = self.encoder_project(current_depths.detach())
            input_c = torch.cat([input_features, context], dim=1)

            gru_hidden = self.gru(gru_hidden, input_c)
            pred_prob = self.p_head(gru_hidden)

            # Metric
            depth_r = (pred_prob * current_depths.detach()).sum(1, keepdim=True)
            pred_depths_r_list.append(depth_r)

            uncertainty_map = torch.sqrt(
                (pred_prob * ((current_depths.detach() - depth_r.repeat(1, bin_num, 1, 1))**2)).sum(1, keepdim=True))
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
        for i in range(max_tree_depth):
            pred_depths_r_list[i] = upsample(pred_depths_r_list[i], scale_factor=4, upsample_type=self.upsample_type,
                                             uncertainty=False)
            pred_depths_c_list[i] = upsample(pred_depths_c_list[i], scale_factor=4, upsample_type=self.upsample_type,
                                             uncertainty=False)
            uncertainty_maps_list[i] = upsample(uncertainty_maps_list[i], scale_factor=4,
                                                upsample_type=self.upsample_type, uncertainty=True)

        result["pred_depths_r_list"] = pred_depths_r_list
        result["pred_depths_c_list"] = pred_depths_c_list
        result["uncertainty_maps_list"] = uncertainty_maps_list

        return result


"""
Global scale per image
"""


class GlobalScale(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, bins_scale=50,
                 virtual_depth_variation=0, upsample_type=1, bins_type=1, bins_type_scale=1):
        super(GlobalScale, self).__init__()
        self.loss_type = loss_type
        self.bins_scale = bins_scale
        self.virtual_depth_variation = virtual_depth_variation
        self.upsample_type = upsample_type

        self.bins_type = bins_type
        self.bins_type_scale = bins_type_scale
        if(self.bins_type == 0):
            print("GlobalScale: uniform bins for canonical\n")
        else:
            print("GlobalScale: log bins for canonical\n")

        if(self.bins_type_scale == 0):
            print("GlobalScale: uniform bins for scale\n")
        else:
            print("GlobalScale: log bins for scale\n")

        if self.virtual_depth_variation == 0:
            self.c_head = PHead(hidden_dim, hidden_dim, bin_num) # Propabilities canonical
            self.s_head = SHead(hidden_dim, num_bins=bins_scale) # Bins scale
            print("GlobalScale: bins for both scale and shift\n") 
        elif self.virtual_depth_variation == 1:
            self.c_head = PHead(hidden_dim, hidden_dim, bin_num) # Propabilities canonical
            self.s_head = SSHead(hidden_dim, 1) # Regression scale 
            print("GlobalScale: bins for canonical\n") 
        elif self.virtual_depth_variation == 2:
            self.c_head = RegressionHead(hidden_dim, hidden_dim) # Regression canonical
            self.s_head = SHead(hidden_dim, num_bins=bins_scale) # Bins scale 
            print("GlobalScale: bins for scale\n") 
        else:
            self.c_head = RegressionHead(hidden_dim, hidden_dim) # Regression canonical
            self.s_head = SSHead(hidden_dim, 1) # Regression scale 
            print("GlobalScale: regression\n") 

        self.relu = nn.ReLU(inplace=True)

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, max_scale=15):
        # Results list #
        pred_depths_r_list = []  # Metric
        pred_depths_rc_list = []  # Canonical
        pred_scale_list = []
        uncertainty_maps_list = []
        pred_depths_c_list = []
        pred_depths_scale_c_list = []
        uncertainty_maps_scale_list = []

        # Canonical bins #
        if self.virtual_depth_variation == 0 or self.virtual_depth_variation == 1:
            if self.bins_type == 0:
                bins_map, bin_edges = get_uniform_bins(depth, min_depth, max_depth, bin_num, sid=False)
            else:
                bins_map, bin_edges = get_uniform_bins(depth, min_depth, max_depth, bin_num, sid=True)

        # Scale bins #
        if self.virtual_depth_variation == 0 or self.virtual_depth_variation == 2:
            if self.bins_type_scale == 0:
                bins_map_scale, bin_scale_edges = get_uniform_bins(
                    torch.zeros(depth.shape[0], 1, 1, 1).to(depth.device), 0, max_scale, self.bins_scale, sid=False)
            else:
                bins_map_scale, bin_scale_edges = get_uniform_bins(
                    torch.zeros(depth.shape[0], 1, 1, 1).to(depth.device), 0, max_scale, self.bins_scale, sid=True)

        # Canonical
        pred_rc = self.c_head(input_feature_map)
        if self.virtual_depth_variation == 0 or self.virtual_depth_variation == 1: # Bins
            # Canonical
            depth_rc = (pred_rc * bins_map.detach()).sum(1, keepdim=True)
            uncertainty_map = torch.sqrt(
                (pred_rc * ((bins_map.detach() - depth_rc.repeat(1, bin_num, 1, 1))**2)).sum(1, keepdim=True))
            
            # Label
            pred_label = get_label(torch.squeeze(depth_rc, 1), bin_edges, bin_num).unsqueeze(1)
            depth_c = torch.gather(bins_map.detach(), 1, pred_label.detach())

            # Upsample #
            depth_rc = upsample(depth_rc, scale_factor=4, upsample_type=self.upsample_type)
            uncertainty_map = upsample(uncertainty_map, scale_factor=4, upsample_type=self.upsample_type, uncertainty=True)
            depth_c = upsample(depth_c, scale_factor=4, upsample_type=self.upsample_type)
        else:
            depth_rc = upsample(pred_rc, scale_factor=4, upsample_type=self.upsample_type)
            uncertainty_map, depth_c = torch.zeros_like(depth_rc), torch.zeros_like(depth_rc)

        pred_depths_rc_list.append(depth_rc)
        uncertainty_maps_list.append(uncertainty_map)
        pred_depths_c_list.append(depth_c)
    
        # Scale 
        pred_scale = self.s_head(input_feature_map)
        if self.virtual_depth_variation == 0 or self.virtual_depth_variation == 2: # Bins
            scale = (pred_scale * bins_map_scale.squeeze(-1).squeeze(-1).detach()).sum(1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
            uncertainty_map = torch.sqrt(
                (pred_scale.unsqueeze(-1).unsqueeze(-1) * ((bins_map_scale.detach() - scale.repeat(1, self.bins_scale, 1, 1))**2)).sum(
                    1, keepdim=True)).squeeze(-1).squeeze(-1)
            
            # Copy to scale bins the result #
            pred_scale = scale.squeeze(-1).squeeze(-1)
            
            # Label
            pred_label = get_label(torch.squeeze(scale, 1), bin_scale_edges, self.bins_scale).unsqueeze(1)
            depth_scale_c = torch.gather(bins_map_scale.detach(), 1, pred_label.detach())
        else:
            uncertainty_map, depth_scale_c = torch.zeros_like(depth_c), torch.zeros_like(depth_c)

        pred_scale_list.append(pred_scale)
        pred_depths_scale_c_list.append(depth_scale_c)
        uncertainty_maps_scale_list.append(uncertainty_map)

        # Metric depth
        if self.loss_type == 0:
            depth_r = (self.relu(depth_rc * pred_scale_list[-1].unsqueeze(1).unsqueeze(1))).clamp(min=1e-3)
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

        return result


"""
Per-Class variations
"""


class PerClassScale(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=5, bins_scale=50,
                 virtual_depth_variation=0, upsample_type=1, bins_type=1, bins_type_scale=1, concat_masks=False):
        super(PerClassScale, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.loss_type = loss_type
        self.bins_scale = bins_scale
        self.virtual_depth_variation = virtual_depth_variation
        self.upsample_type = upsample_type

        self.concat_masks = concat_masks

        self.bins_type = bins_type
        self.bins_type_scale = bins_type_scale

        if(self.bins_type == 0):
            print("PerClassScale: uniform bins for canonical\n")
        else:
            print("PerClassScale: log bins for canonical\n")

        if(self.bins_type_scale == 0):
            print("PerClassScale: uniform bins for scale\n")
        else:
            print("PerClassScale: log bins for scale\n")

        if self.concat_masks:
            in_dim = 128 + 1
        else:
            in_dim = 128

        self.p_heads = nn.ModuleList()
        self.s_heads = nn.ModuleList()
        for i in range(num_semantic_classes):
            if self.virtual_depth_variation == 0: # Propabilities both scale and canonical
                self.p_heads.append(PHead(in_idm, 128, bin_num=bin_num))
                self.s_heads.append(SHead(in_dim, num_bins=self.bins_scale))
            elif self.virtual_depth_variation == 1: # Probabilities canonical 
                self.p_heads.append(PHead(in_dim, 128, bin_num=bin_num))
                self.s_heads.append(SSHead(in_dim, num_out=1))
            elif self.virtual_depth_variation == 2: # Probabilities scale 
                self.p_heads.append(SigmoidHead(in_dim, 128, num_classes=1))
                self.s_heads.append(SHead(in_dim, num_bins=self.bins_scale))
            else: # Regression 
                self.p_heads.append(SigmoidHead(in_dim, 128, num_classes=1))
                self.s_heads.append(SSHead(in_dim, num_out=1))

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, masks, min_scale=0,max_scale=15):
        pred_depths_r_list = []  # Metric
        pred_depths_rc_list = []  # Canonical

        pred_scale_list = []

        uncertainty_maps_list = []
        uncertainty_maps_scale_list = []
        pred_depths_c_list = []
        pred_depths_scale_c_list = []

        b, _, h, w = depth.shape
        h *= 4
        w *= 4
        depth_rc = []
        pred_scale = []

        # Canonical bins #
        if self.virtual_depth_variation == 0 or self.virtual_depth_variation == 1:
            if self.bins_type == 0:
                bins_map, bin_edges = get_uniform_bins(depth, min_depth, max_depth, bin_num, sid=False)
            else:
                bins_map, bin_edges = get_uniform_bins(depth, min_depth, max_depth, bin_num, sid=True)

        # Scale bins #
        if self.virtual_depth_variation == 0 or self.virtual_depth_variation == 2:
            if self.bins_type_scale == 0:
                bins_map_scale, bin_scale_edges = get_uniform_bins(
                    torch.zeros(depth.shape[0], 1, 1, 1).to(depth.device), min_scale, max_scale, self.bins_scale, sid=False)
            else: 
                bins_map_scale, bin_scale_edges = get_uniform_bins(
                    torch.zeros(depth.shape[0], 1, 1, 1).to(depth.device), min_scale, max_scale, self.bins_scale, sid=True)

        uncertainty_map = torch.zeros(b, self.num_semantic_classes, h, w).to(masks.device)
        depth_c = torch.zeros(b, self.num_semantic_classes, h, w).to(masks.device)

        uncertainty_map_scale = torch.zeros(b, self.num_semantic_classes, 1, 1).to(masks.device)
        c_map_scale = torch.zeros(b, self.num_semantic_classes, 1, 1).to(masks.device)

        # Predict canonical and scale per class
        for i in range(self.num_semantic_classes):

            # Concat mask #
            if self.concat_masks:
                input_feature_map_current = torch.cat((input_feature_map, masks[:, i, :, :].unsqueeze(1)), dim=1)
            else:
                input_feature_map_current = input_feature_map

            # Canonical
            prob = self.p_heads[i](input_feature_map_current)
            if self.virtual_depth_variation == 0 or self.virtual_depth_variation == 1:
                # Canonical depth estimation #
                depth_rc_current = (prob * bins_map.detach()).sum(1, keepdim=True)
             
                # Uncertainty #
                uncertainty_map_current = torch.sqrt(
                    (prob * ((bins_map.detach() - depth_rc_current.repeat(1, bin_num, 1, 1))**2)).sum(1, keepdim=True))
                
                # Labels #
                pred_label = get_label(torch.squeeze(depth_rc_current, 1), bin_edges, bin_num).unsqueeze(1)
                depth_c_current = torch.gather(bins_map.detach(), 1, pred_label.detach())          

                # Upsample #
                depth_rc_current = upsample(depth_rc_current, scale_factor=4, upsample_type=self.upsample_type)
                depth_c_current = upsample(depth_c_current, scale_factor=4, upsample_type=self.upsample_type)
                uncertainty_map_current = upsample(uncertainty_map_current, scale_factor=4, upsample_type=self.upsample_type, uncertainty=True)
            
                depth_rc.append(depth_rc_current)
                depth_c[:, i, :, :] = depth_c_current.squeeze(1)
                uncertainty_map[:, i, :, :] = uncertainty_map_current.squeeze(1)
            else:
                prob = upsample(prob, scale_factor=4, upsample_type=self.upsample_type)
                depth_rc_current = prob
                
                depth_rc.append(depth_rc_current)
                uncertainty_map[:, i, :, :] = torch.zeros_like(uncertainty_map[:, i, :, :])
                depth_c[:, i, :, :] = torch.zeros_like(depth_c[:, i, :, :])
             

            # Scale
            scale_current = self.s_heads[i](input_feature_map_current)
            # Bins #
            if self.virtual_depth_variation == 0 or self.virtual_depth_variation == 2:
                # Estimate scale #
                scale_current = scale_current.unsqueeze(-1).unsqueeze(-1)
                scale = (scale_current * bins_map_scale.detach()).sum(1, keepdim=True)
                pred_scale.append(scale.squeeze(-1).squeeze(-1))

                # Uncertainty #
                uncertainty_map_current_scale = torch.sqrt(
                    (scale_current * ((bins_map_scale.detach() - scale)**2)).sum(1, keepdim=True))
                uncertainty_map_scale[:, i, :, :] = uncertainty_map_current_scale.squeeze(1)
                
                # Labels #
                pred_label = get_label(torch.squeeze(scale, 1), bin_scale_edges, self.bins_scale).unsqueeze(1)
                depth_scale_c = torch.gather(bins_map_scale.detach(), 1, pred_label.detach())
                c_map_scale[:, i, :, :] = depth_scale_c.squeeze(1)
            else:
                pred_scale.append(scale_current)
                uncertainty_map_scale[:, i, :, :] = torch.zeros_like(uncertainty_map_scale[:, i, :, :])
                c_map_scale[:, i, :, :] = torch.zeros_like(c_map_scale[:, i, :, :])

        depth_rc = torch.cat(depth_rc, dim=1)

        # Save #
        pred_depths_c_list.append(depth_c)
        pred_depths_rc_list.append(depth_rc)
        uncertainty_maps_list.append(uncertainty_map)

        pred_scale = torch.cat(pred_scale, dim=1)
        pred_scale_list.append(pred_scale)
        
        # Metric depth
        if self.loss_type == 0:
            depth_r = (self.relu(depth_rc * pred_scale_list[-1].unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
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

        return result


"""
Instances variations 
"""


class UniformInstancesSharedCanonical(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=14, \
                 num_instances=63,var=0, padding_instances=0, roi_align=0, roi_align_size=32, bins_scale=150, virtual_depth_variation=0, upsample_type=1):
        super(UniformInstancesSharedCanonical, self).__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.loss_type = loss_type
        self.bins_scale = bins_scale

        self.num_semantic_classes = num_semantic_classes
        self.num_instances = num_instances
        self.var = var  # Variation
        self.virtual_depth_variation = virtual_depth_variation
        self.upsample_type = upsample_type

        global padding_global

        self.padding_instances = padding_instances
        padding_global = padding_instances

        if self.virtual_depth_variation == 0:
            self.instances_scale_and_shift = ROISelectScale(128, downsampling=4,
                                                            num_out=2 * (self.num_semantic_classes))
        elif self.virtual_depth_variation == 1:
            self.instances_scale_and_shift = ROISelectScale(128, downsampling=4, num_out=self.num_semantic_classes)
        elif self.virtual_depth_variation == 2:
            self.instances_scale_and_shift = ROISelectScaleBins(128, downsampling=4,
                                                                num_semantic_classes=self.num_semantic_classes,
                                                                num_bins=self.bins_scale)

        self.instances_canonical = ROISelectSharedCanonicalUniform(128, bin_num=bin_num)

        self.roi_align = roi_align
        if roi_align == 1:
            output_size = (roi_align_size, roi_align_size)
            spatial_scale = 1.0 / 4
            sampling_ratio = 4

            self.roiAlign = RoIAlign(output_size, spatial_scale=spatial_scale, sampling_ratio=sampling_ratio)

        # Pick variation #
        if var == 1:
            self.instances_canonical = ROISelectSharedCanonicalBigUniform(128, bin_num=bin_num)
        if var == 2:
            self.instances_canonical = ROISelectSharedCanonicalHugeUniform(128, bin_num=bin_num)
        if var == 3:
            if self.virtual_depth_variation == 0:
                self.instances_scale_and_shift = ROISelectScaleBig(128, downsampling=4,
                                                                   num_out=2 * (self.num_semantic_classes))
            elif self.virtual_depth_variation == 1:
                self.instances_scale_and_shift = ROISelectScaleBig(128, downsampling=4,
                                                                   num_out=self.num_semantic_classes)
            elif self.virtual_depth_variation == 2:
                self.instances_scale_and_shift = ROISelectScaleBigBins(128, downsampling=4,
                                                                       num_semantic_classes=self.num_semantic_classes,
                                                                       num_bins=self.bins_scale)
        if var == 4:
            if self.virtual_depth_variation == 0:
                self.instances_scale_and_shift = ROISelectScaleHuge(128, downsampling=4,
                                                                    num_out=2 * (self.num_semantic_classes))
            elif self.virtual_depth_variation == 1:
                self.instances_scale_and_shift = ROISelectScaleHuge(128, downsampling=4,
                                                                    num_out=self.num_semantic_classes)
            elif self.virtual_depth_variation == 2:
                self.instances_scale_and_shift = ROISelectScaleHugeBins(128, downsampling=4,
                                                                        num_semantic_classes=self.num_semantic_classes,
                                                                        num_bins=self.bins_scale)
        if var == 5:
            if self.virtual_depth_variation == 0:
                self.instances_scale_and_shift = ROISelectScaleSmall(128, downsampling=4,
                                                                     num_out=2 * (self.num_semantic_classes))
            elif self.virtual_depth_variation == 1:
                self.instances_scale_and_shift = ROISelectScaleSmall(128, downsampling=4,
                                                                     num_out=self.num_semantic_classes)
            elif self.virtual_depth_variation == 2:
                self.instances_scale_and_shift = ROISelectScaleSmallBins(128, downsampling=4,
                                                                         num_semantic_classes=self.num_semantic_classes,
                                                                         num_bins=self.bins_scale)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, masks, instances, boxes,
                labels):
        pred_depths_instances_r_list = []  # Metric
        pred_depths_instances_rc_list = []  # Canonical

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
            boxes_roi = [(boxes_roi[instances_per_batch[i]:instances_per_batch[i + 1]])
                         for i in range(instances_per_batch.shape[0] - 1)]

            input_feature_map_instances_roi = self.roiAlign(input_feature_map, boxes_roi)

        if self.virtual_depth_variation == 0:
            instances_scale_shift = self.instances_scale_and_shift(input_feature_map_instances_roi, boxes, labels)
            instances_scale, instances_shift = pick_predictions_instances_scale_shift(instances_scale_shift, labels)
            pred_scale_instances_list.append(instances_scale)
            pred_shift_instances_list.append(instances_shift)
        elif self.virtual_depth_variation == 1:
            instances_scale_shift = self.instances_scale_and_shift(input_feature_map_instances_roi, boxes, labels)
            instances_scale, instances_shift = pick_predictions_instances_scale_shift(
                instances_scale_shift, labels, only_scale=True)
            pred_scale_instances_list.append(instances_scale)
        elif self.virtual_depth_variation == 2:
            instances_scale_unc = self.instances_scale_and_shift(input_feature_map_instances_roi, boxes, labels)
            instances_scale, instances_scale_unc = pick_predictions_instances_scale_shift(instances_scale_unc, labels)
            pred_scale_instances_list.append(instances_scale)
            uncertainty_maps_scale_list.append(instances_scale_unc)

        # Canonical instances valid #
        instances_canonical_prob = self.instances_canonical(input_feature_map, boxes, labels)

        bins_map, bin_edges = get_uniform_bins(depth, min_depth, max_depth, bin_num)
        bins_map = bins_map[0, :, :, :].unsqueeze(0)

        bins_map = torch.cat([bins_map] * instances_canonical_prob.shape[0], dim=0)
        depth_instances_rc = (instances_canonical_prob * bins_map.detach()).sum(1, keepdim=True)

        # Unc #
        uncertainty_map_current = torch.sqrt(
            (instances_canonical_prob * ((bins_map.detach() - depth_instances_rc.repeat(1, bin_num, 1, 1))**2)).sum(
                1, keepdim=True))
        uncertainty_map = uncertainty_map_current

        # Labels #
        pred_label = get_label(torch.squeeze(depth_instances_rc, 1), bin_edges, bin_num).unsqueeze(1)
        depth_c = (torch.gather(bins_map.detach(), 1, pred_label.detach()))

        # Fill full instances map canonical #
        canonical_full = torch.zeros((batch_size * max_instances_size, 1, h, w)).to(depth_instances_rc.device)
        uncertainty_maps_full = torch.zeros((batch_size * max_instances_size, 1, h, w)).to(depth_instances_rc.device)
        depth_c_full = torch.zeros((batch_size * max_instances_size, 1, h, w)).to(depth_instances_rc.device)

        valid_boxes = labels.view(batch_size * max_instances_size, 1)
        valid_boxes = torch.nonzero(valid_boxes != -1)

        canonical_full[valid_boxes[:, 0]] = depth_instances_rc
        depth_instances_rc = canonical_full.view(batch_size, max_instances_size, h, w)

        uncertainty_maps_full[valid_boxes[:, 0]] = uncertainty_map
        uncertainty_map = uncertainty_maps_full.view(batch_size, max_instances_size, h, w)

        depth_c_full[valid_boxes[:, 0]] = depth_c
        depth_c = depth_c_full.view(batch_size, max_instances_size, h, w)

        # Upsample #
        depth_instances_rc = upsample(depth_instances_rc, scale_factor=4, upsample_type=self.upsample_type)
        uncertainty_map = upsample(uncertainty_map, scale_factor=4, upsample_type=self.upsample_type, uncertainty=True)
        depth_c = upsample(depth_c, scale_factor=4, upsample_type=self.upsample_type)

        pred_depths_instances_rc_list.append(depth_instances_rc)
        uncertainty_maps_list.append(uncertainty_map)
        pred_depths_c_list.append(depth_c)

        # Metric
        if self.loss_type == 0:
            if self.virtual_depth_variation == 0:
                depth_instances_r = (self.relu(depth_instances_rc * instances_scale.unsqueeze(-1).unsqueeze(-1) +
                                               instances_shift.unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
            else:
                depth_instances_r = (self.relu(depth_instances_rc *
                                               instances_scale.unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
        else:
            if self.virtual_depth_variation == 0:
                depth_instances_r = depth_instances_rc * instances_scale.unsqueeze(-1).unsqueeze(
                    -1) + instances_shift.unsqueeze(-1).unsqueeze(-1)
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

        self.instances_scale_and_shift = ROISelectScale(128, downsampling=4,
                                                        num_semantic_classes=self.num_semantic_classes)
        self.instances_canonical = ROISelectCanonical(128, 4, num_semantic_classes=self.num_semantic_classes)

        # Pick variation #
        if var == 1:
            self.instances_scale_and_shift = ROISelectScaleA(128, downsampling=4,
                                                             num_semantic_classes=self.num_semantic_classes)
            self.instances_canonical = ROISelectCanonical(128, 4, num_semantic_classes=self.num_semantic_classes)
        if var == 2:
            self.instances_scale_and_shift = ROISelectScaleB(128, downsampling=4,
                                                             num_semantic_classes=self.num_semantic_classes)
            self.instances_canonical = ROISelectCanonical(128, 4, num_semantic_classes=self.num_semantic_classes)
        if var == 3:
            self.instances_scale_and_shift = ROISelectScale(128, downsampling=4,
                                                            num_semantic_classes=self.num_semantic_classes)
            self.instances_canonical = ROISelectCanonicalA(128, num_semantic_classes=self.num_semantic_classes)
        if var == 4:
            self.instances_scale_and_shift = ROISelectScale(128, downsampling=4,
                                                            num_semantic_classes=self.num_semantic_classes)
            self.instances_canonical = ROISelectCanonicalB(128, num_semantic_classes=self.num_semantic_classes)
        if var == 5:
            self.instances_scale_and_shift = ROISelectScale(128, downsampling=4,
                                                            num_semantic_classes=self.num_semantic_classes)
            self.instances_canonical = ROISelectCanonicalC(128, num_semantic_classes=self.num_semantic_classes)
        if var == 6:
            self.instances_scale_and_shift = ROISelectScaleB(128, downsampling=4,
                                                             num_semantic_classes=self.num_semantic_classes)
            self.instances_canonical = ROISelectCanonicalC(128, num_semantic_classes=self.num_semantic_classes)
        if var == 7:
            self.instances_scale_and_shift = ROISelectScaleA(128, downsampling=4,
                                                             num_semantic_classes=self.num_semantic_classes)
            self.instances_canonical = ROISelectCanonicalC(128, num_semantic_classes=self.num_semantic_classes)
        if var == 8:
            self.instances_scale_and_shift = ROISelectScaleC(128, downsampling=4,
                                                             num_semantic_classes=self.num_semantic_classes)
            self.instances_canonical = ROISelectCanonical(128, 4, num_semantic_classes=self.num_semantic_classes)
        if var == 9:
            self.instances_scale_and_shift = ROISelectScale(128, downsampling=4,
                                                            num_semantic_classes=self.num_semantic_classes)
            self.instances_canonical = ROISelectCanonicalD(128, num_semantic_classes=self.num_semantic_classes)
        if var == 10:
            self.instances_scale_and_shift = ROISelectScaleC(128, downsampling=4,
                                                             num_semantic_classes=self.num_semantic_classes)
            self.instances_canonical = ROISelectCanonicalD(128, num_semantic_classes=self.num_semantic_classes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, masks, instances, boxes,
                labels):
        pred_depths_instances_r_list = []  # Metric
        pred_depths_instances_rc_list = []  # Canonical

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
            depth_instances_r = (self.relu(instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) +
                                           instances_shift.unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
        else:
            depth_instances_r = instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(
                -1) + instances_shift.unsqueeze(-1).unsqueeze(-1)

        pred_depths_instances_r_list.append(depth_instances_r)

        result = {}
        result["pred_depths_instances_r_list"] = pred_depths_instances_r_list
        result["pred_depths_instances_rc_list"] = pred_depths_instances_rc_list
        result["pred_scale_instances_list"] = pred_scale_instances_list
        result["pred_shift_instances_list"] = pred_shift_instances_list

        return result


class RegressionInstancesSharedCanonical(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, loss_type=0, num_semantic_classes=14, num_instances=63, var=0,
                 padding_instances=0):
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

        self.instances_scale_and_shift = ROISelectScale(128, downsampling=4, num_out=2 * (self.num_semantic_classes))
        self.instances_canonical = ROISelectSharedCanonical(128, num_semantic_classes=1)

        # Pick variation #
        if var == 1:
            self.instances_canonical = ROISelectSharedCanonicalBig(128, num_semantic_classes=1)
        if var == 2:
            self.instances_canonical = ROISelectSharedCanonicalHuge(128, num_semantic_classes=1)
        if var == 3:
            self.instances_scale_and_shift = ROISelectScaleBig(128, downsampling=4,
                                                               num_out=2 * (self.num_semantic_classes))
        if var == 4:
            self.instances_scale_and_shift = ROISelectScaleHuge(128, downsampling=4,
                                                                num_out=2 * (self.num_semantic_classes))
        if var == 5:
            self.instances_scale_and_shift = ROISelectScaleSmall(128, downsampling=4,
                                                                 num_out=2 * (self.num_semantic_classes))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, masks, instances, boxes,
                labels):
        pred_depths_instances_r_list = []  # Metric
        pred_depths_instances_rc_list = []  # Canonical

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

        canonical_full = torch.zeros((batch_size * num_max_instances, 1, h, w)).to(instances_canonical.device)
        canonical_full[boxes_valid_idx[:, 0]] = instances_canonical

        canonical_full = canonical_full.view(batch_size, num_max_instances, h, w)
        instances_canonical = canonical_full
        pred_depths_instances_rc_list.append(instances_canonical)

        # Metric
        if self.loss_type == 0:
            depth_instances_r = (self.relu(instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) +
                                           instances_shift.unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
        else:
            if self.var == 8:
                depth_instances_r = instances_canonical * 50 * F.sigmoid(
                    instances_scale.unsqueeze(-1).unsqueeze(-1)) + instances_shift.unsqueeze(-1).unsqueeze(-1)

            elif self.var == 9 or self.var == 13 or self.var == 14 or self.var == 15:
                depth_instances_r = instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(
                    -1) + instances_shift.unsqueeze(-1).unsqueeze(-1)

            else:
                depth_instances_r = instances_canonical * 20 * F.sigmoid(
                    instances_scale.unsqueeze(-1).unsqueeze(-1)) + instances_shift.unsqueeze(-1).unsqueeze(-1)

        pred_depths_instances_r_list.append(depth_instances_r)

        result = {}
        result["pred_depths_instances_r_list"] = pred_depths_instances_r_list
        result["pred_depths_instances_rc_list"] = pred_depths_instances_rc_list
        result["pred_scale_instances_list"] = pred_scale_instances_list
        result["pred_shift_instances_list"] = pred_shift_instances_list

        return result


class RegressionInstancesSharedCanonicalClass(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, loss_type=0, num_semantic_classes=14, num_instances=63, var=0,
                 padding_instances=0):
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

        self.instances_scale_and_shift = ROISelectScale(128, downsampling=4, num_out=2 * (self.num_semantic_classes))
        self.instances_canonical = ROISelectCanonicalSharedClass(128, num_semantic_classes=self.num_semantic_classes)

        # Pick variation #
        if var == 1 or var == 3:
            self.instances_scale_and_shift = ROISelectScaleBig(128, downsampling=4,
                                                               num_out=2 * (self.num_semantic_classes))
        if var == 2 or var == 3:
            self.instances_canonical = ROISelectCanonicalSharedClassBig(128,
                                                                        num_semantic_classes=self.num_semantic_classes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, masks, instances, boxes,
                labels):
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
        input_feature_map_instances_roi_canonical = roi_select_features_canonical_shared(
            input_feature_map_instances, boxes, labels, num_semantic_classes=self.num_semantic_classes)

        input_feature_map_instances_roi_canonical = input_feature_map_instances_roi_canonical.view(
            batch_size * (self.num_semantic_classes - 1), hidden_dim, height_hidden, width_hidden)
        instances_canonical = self.instances_canonical(input_feature_map_instances_roi_canonical)

        instances_canonical = pick_predictions_instances_canonical_shared_class(instances_canonical, labels)
        pred_depths_instances_rc_list.append(instances_canonical)

        # Metric
        if self.loss_type == 0:
            depth_instances_r = (self.relu(instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) +
                                           instances_shift.unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
        else:
            if self.var == 8:
                depth_instances_r = instances_canonical * 50 * F.sigmoid(
                    instances_scale.unsqueeze(-1).unsqueeze(-1)) + instances_shift.unsqueeze(-1).unsqueeze(-1)

            elif self.var == 9 or self.var == 13 or self.var == 14 or self.var == 15:
                depth_instances_r = instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(
                    -1) + instances_shift.unsqueeze(-1).unsqueeze(-1)

            else:
                depth_instances_r = instances_canonical * 20 * F.sigmoid(
                    instances_scale.unsqueeze(-1).unsqueeze(-1)) + instances_shift.unsqueeze(-1).unsqueeze(-1)

        pred_depths_instances_r_list.append(depth_instances_r)

        result = {}
        result["pred_depths_instances_r_list"] = pred_depths_instances_r_list
        result["pred_depths_instances_rc_list"] = pred_depths_instances_rc_list
        result["pred_scale_instances_list"] = pred_scale_instances_list
        result["pred_shift_instances_list"] = pred_shift_instances_list

        return result


class RegressionInstancesSharedCanonicalModuleScale(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, loss_type=0, num_semantic_classes=14, num_instances=63, var=0,
                 padding_instances=0):
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
                self.instances_scale_and_shift.append(
                    ROISelectScaleModuleBig(128, downsampling=4, num_semantic_classes=1))
        if var == 2 or var == 3:
            self.instances_canonical = ROISelectSharedCanonicalBig(128, num_semantic_classes=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, masks, instances, boxes,
                labels):
        pred_depths_instances_r_list = []  # Metric
        pred_depths_instances_rc_list = []  # Canonical

        pred_scale_instances_list = []
        pred_shift_instances_list = []

        batch_size, num_max_instances, h, w = instances.shape

        labels_reshaped = labels.view(batch_size * num_max_instances, 1)

        instances_scale = torch.zeros((batch_size * num_max_instances, 1)).to(labels.device)
        instances_shift = torch.zeros((batch_size * num_max_instances, 1)).to(labels.device)

        # Pass class-specific feature maps to appropriate scale head #
        for i in range(self.num_semantic_classes):
            with torch.no_grad():
                valid_boxes_class = torch.nonzero(labels_reshaped == i)

                if valid_boxes_class.shape[0] == 0:
                    continue

            input_feature_map_instances_roi = roi_select_features_module(input_feature_map, boxes, labels, i)
            scale_shift = self.instances_scale_and_shift[i](input_feature_map_instances_roi, boxes, labels, i)

            instances_scale[valid_boxes_class[:, 0]] = scale_shift[:, ::2]
            instances_shift[valid_boxes_class[:, 0]] = scale_shift[:, 1::2]

        instances_scale = instances_scale.view(batch_size, num_max_instances)
        instances_shift = instances_shift.view(batch_size, num_max_instances)

        pred_scale_instances_list.append(instances_scale)
        pred_shift_instances_list.append(instances_shift)

        # Canonical shared #
        instances_canonical = self.instances_canonical(input_feature_map, boxes, labels)

        # Copy the shared canonical to all instances #
        valid_boxes = torch.nonzero(labels_reshaped != -1)
        canonical_full = torch.zeros((batch_size * num_max_instances, 1, h, w)).to(instances_canonical.device)

        canonical_full[valid_boxes[:, 0]] = instances_canonical
        canonical_full = canonical_full.view(batch_size, num_max_instances, h, w)
        instances_canonical = canonical_full
        pred_depths_instances_rc_list.append(instances_canonical)

        # Metric
        if self.loss_type == 0:
            depth_instances_r = (self.relu(instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) +
                                           instances_shift.unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
        else:
            depth_instances_r = instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(
                -1) + instances_shift.unsqueeze(-1).unsqueeze(-1)

        pred_depths_instances_r_list.append(depth_instances_r)

        result = {}

        result["pred_depths_instances_r_list"] = pred_depths_instances_r_list
        result["pred_depths_instances_rc_list"] = pred_depths_instances_rc_list
        result["pred_scale_instances_list"] = pred_scale_instances_list
        result["pred_shift_instances_list"] = pred_shift_instances_list

        return result


class RegressionInstancesModule(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, loss_type=0, num_semantic_classes=14, num_instances=63, var=0,
                 padding_instances=0):
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
                self.instances_scale_and_shift.append(
                    ROISelectScaleModuleBig(128, downsampling=4, num_semantic_classes=1))
        if var == 2 or var == 3:
            for i in range(num_semantic_classes - 1):
                self.instances_canonical.append(ROISelectCanonicalModuleBig(128, num_semantic_classes=1))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, masks, instances, boxes,
                labels):
        pred_depths_instances_r_list = []  # Metric
        pred_depths_instances_rc_list = []  # Canonical

        pred_scale_instances_list = []
        pred_shift_instances_list = []
        b, _, h, w = depth.shape

        batch_size, num_max_instances, h, w = instances.shape

        labels_reshaped = labels.view(batch_size * num_max_instances, 1)

        instances_scale = torch.zeros((batch_size * num_max_instances, 1)).to(labels.device)
        instances_shift = torch.zeros((batch_size * num_max_instances, 1)).to(labels.device)
        canonical_full = torch.zeros((batch_size * num_max_instances, 1, h, w)).to(labels.device)

        # Pass class-specific feature maps to appropriate scale and canonical heads #
        for i in range(self.num_semantic_classes):
            with torch.no_grad():
                valid_boxes_class = torch.nonzero(labels_reshaped == i)

                if valid_boxes_class.shape[0] == 0:
                    continue

            # Scale/Shift #
            input_feature_map_instances_roi = roi_select_features_module(input_feature_map, boxes, labels, i)
            scale_shift = self.instances_scale_and_shift[i](input_feature_map_instances_roi, boxes, labels, i)
            instances_scale[valid_boxes_class[:, 0]] = scale_shift[:, ::2]
            instances_shift[valid_boxes_class[:, 0]] = scale_shift[:, 1::2]

            # Canonical #
            instances_canonical = self.instances_canonical[i](input_feature_map_instances_roi, boxes, labels, i)
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
            depth_instances_r = (self.relu(instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(-1) +
                                           instances_shift.unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-3)
        else:
            depth_instances_r = instances_canonical * instances_scale.unsqueeze(-1).unsqueeze(
                -1) + instances_shift.unsqueeze(-1).unsqueeze(-1)

        pred_depths_instances_r_list.append(depth_instances_r)

        result = {}
        result["pred_depths_instances_r_list"] = pred_depths_instances_r_list
        result["pred_depths_instances_rc_list"] = pred_depths_instances_rc_list
        result["pred_scale_instances_list"] = pred_scale_instances_list
        result["pred_shift_instances_list"] = pred_shift_instances_list

        return result
