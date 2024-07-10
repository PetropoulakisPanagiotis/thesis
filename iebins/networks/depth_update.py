import torch
import torch.nn as nn
from torchvision.ops import RoIAlign

from .depth_update_heads import *
from .utils import *
"""
IEBins implementation 
"""


class IEBINS(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, upsample_type=0):
        super(IEBINS, self).__init__()
        self.upsample_type = upsample_type

        self.encoder_project = ProjectionInputDepth(hidden_dim=hidden_dim, out_chs=hidden_dim * 2, bin_num=bin_num)

        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=self.encoder_project.out_chs + context_dim)

        self.p_head = ProbCanonicalHead(hidden_dim, hidden_dim, bin_num)

    def forward(self, depth, context, gru_hidden, bin_num, min_depth, max_depth, max_tree_depth=6):

        # Depth #
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

            # Update bin canditates with target bin and uncertainty #
            bin_edges, current_depths = update_bins(bin_edges, target_bin_left, target_bin_right, depth_r.detach(), pred_label.detach(), \
                                                    bin_num, min_depth, max_depth, uncertainty_map)

        # Upsample #
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
        self.virtual_depth_variation = virtual_depth_variation
        self.upsample_type = upsample_type

        self.bins_type = bins_type
        self.bins_type_scale = bins_type_scale

        self.bin_num = bin_num
        self.bins_scale = bins_scale

        self.sid_canonical = False if self.bins_type == 0 else True
        self.sid_scale = False if self.bins_type_scale == 0 else True

        if self.bins_type == 0:
            print("GlobalScale: uniform bins for canonical")
        else:
            print("GlobalScale: log bins for canonical")

        if self.bins_type_scale == 0:
            print("GlobalScale: uniform bins for scale")
        else:
            print("GlobalScale: log bins for scale")

        if self.virtual_depth_variation == 0:
            self.c_head = ProbCanonicalHead(hidden_dim, hidden_dim, self.bin_num)  # Propabilities canonical
            self.s_head = ProbScaleHead(hidden_dim, num_bins=self.bins_scale)  # Bins scale
            print("GlobalScale: bins for both scale and shift")
        elif self.virtual_depth_variation == 1:
            self.c_head = ProbCanonicalHead(hidden_dim, hidden_dim, self.bin_num)  # Propabilities canonical
            self.s_head = RegressScaleHead(hidden_dim)  # Regression scale
            print("GlobalScale: bins for canonical")
        elif self.virtual_depth_variation == 2:
            self.c_head = RegressCanonicalHead(hidden_dim, hidden_dim)  # Regression canonical
            self.s_head = ProbScaleHead(hidden_dim, num_bins=self.bins_scale)  # Bins scale
            print("GlobalScale: bins for scale")
        else:
            self.c_head = RegressCanonicalHead(hidden_dim, hidden_dim)  # Regression canonical
            self.s_head = RegressScaleHead(hidden_dim)  # Regression scale
            print("GlobalScale: regression")

        self.relu = nn.ReLU(inplace=True)

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, max_scale=15, max_tree_depth=1):
        # Depth #
        pred_depths_r_list = []  # Metric
        pred_depths_rc_list = []  # Canonical
        pred_depths_c_list = []  # Labels

        # Scale #
        pred_scale_list = []
        pred_depths_scale_c_list = []

        # Uncertainty #
        uncertainty_maps_list = []
        uncertainty_maps_scale_list = []

        # Canonical bins #
        if self.virtual_depth_variation == 0 or self.virtual_depth_variation == 1:
            bins_map, bin_edges = get_uniform_bins(depth, min_depth, max_depth, bin_num, sid=self.sid_canonical)

        # Scale bins #
        if self.virtual_depth_variation == 0 or self.virtual_depth_variation == 2:
            bins_map_scale, bin_scale_edges = get_uniform_bins(
                torch.zeros(depth.shape[0], 1, 1, 1).to(depth.device), 0, max_scale, self.bins_scale,
                sid=self.sid_scale)

        # Canonical #
        # Prediction: either canonical values or probs #
        pred_rc = self.c_head(input_feature_map)
        if self.virtual_depth_variation == 0 or self.virtual_depth_variation == 1:  # Bins
            # Canonical prediction #
            depth_rc = (pred_rc * bins_map.detach()).sum(1, keepdim=True)
            uncertainty_map = torch.sqrt(
                (pred_rc * ((bins_map.detach() - depth_rc.repeat(1, bin_num, 1, 1))**2)).sum(1, keepdim=True))

            # Label
            pred_label = get_label(torch.squeeze(depth_rc, 1), bin_edges, bin_num).unsqueeze(1)
            depth_c = torch.gather(bins_map.detach(), 1, pred_label.detach())

            # Upsample #
            depth_rc = upsample(depth_rc, scale_factor=4, upsample_type=self.upsample_type)
            uncertainty_map = upsample(uncertainty_map, scale_factor=4, upsample_type=self.upsample_type,
                                       uncertainty=True)
            depth_c = upsample(depth_c, scale_factor=4, upsample_type=self.upsample_type)
        else:
            depth_rc = upsample(pred_rc, scale_factor=4, upsample_type=self.upsample_type)
            uncertainty_map, depth_c = torch.zeros_like(depth_rc).clamp(min=1e-4), torch.zeros_like(depth_rc)

        pred_depths_rc_list.append(depth_rc)
        uncertainty_maps_list.append(uncertainty_map)
        pred_depths_c_list.append(depth_c)

        # Scale #
        # Prediction: either scale values or probs #
        pred_scale = self.s_head(input_feature_map)
        if self.virtual_depth_variation == 0 or self.virtual_depth_variation == 2:  # Bins
            scale = (pred_scale * bins_map_scale.squeeze(-1).squeeze(-1).detach()).sum(
                1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
            uncertainty_map = torch.sqrt((pred_scale.unsqueeze(-1).unsqueeze(-1) *
                                          ((bins_map_scale.detach() - scale.repeat(1, self.bins_scale, 1, 1))**2)).sum(
                                              1, keepdim=True)).squeeze(-1).squeeze(-1)

            # Copy to scale bins the result #
            pred_scale = scale.squeeze(-1).squeeze(-1)
            
            # Label #
            pred_label = get_label(torch.squeeze(scale, 1), bin_scale_edges, self.bins_scale).unsqueeze(1)
            depth_scale_c = torch.gather(bins_map_scale.detach(), 1, pred_label.detach())
        else:
            uncertainty_map, depth_scale_c = torch.zeros_like(depth_c).clamp(min=1e-4), torch.zeros_like(depth_c)

        pred_scale_list.append(pred_scale)
        pred_depths_scale_c_list.append(depth_scale_c)
        uncertainty_maps_scale_list.append(uncertainty_map)

        # Metric depth #
        if self.loss_type == 0:
            depth_r = (self.relu(depth_rc * pred_scale_list[-1].unsqueeze(1).unsqueeze(1))).clamp(min=1e-4)
        else:
            depth_r = depth_rc * pred_scale_list[-1].unsqueeze(1).unsqueeze(1)

        pred_depths_r_list.append(depth_r)

        result = {}
        result["pred_depths_r_list"] = pred_depths_r_list
        result["pred_depths_rc_list"] = pred_depths_rc_list
        result["pred_depths_c_list"] = pred_depths_c_list

        result["pred_scale_list"] = pred_scale_list
        result["pred_depths_scale_c_list"] = pred_depths_scale_c_list

        result["uncertainty_maps_list"] = uncertainty_maps_list
        result["uncertainty_maps_scale_list"] = uncertainty_maps_scale_list

        return result


"""
Per-Class scale
"""


class PerClassScale(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=5, bins_scale=50,
                 virtual_depth_variation=0, upsample_type=1, bins_type=1, bins_type_scale=1, concat_masks=False):
        super(PerClassScale, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
       
        self.loss_type = loss_type
        self.concat_masks = concat_masks

        self.virtual_depth_variation = virtual_depth_variation
        self.upsample_type = upsample_type
        
        self.bins_type = bins_type
        self.bins_type_scale = bins_type_scale
        self.bins_scale = bins_scale

        self.sid_canonical = False if self.bins_type == 0 else True
        self.sid_scale = False if self.bins_type_scale == 0 else True

        if self.bins_type == 0:
            print("PerClassScale: uniform bins for canonical")
        else:
            print("PerClassScale: log bins for canonical")

        if self.bins_type_scale == 0:
            print("PerClassScale: uniform bins for scale")
        else:
            print("PerClassScale: log bins for scale")

        if self.concat_masks:
            in_dim = 128 + 1
        else:
            in_dim = 128

        self.p_heads = nn.ModuleList()
        self.s_heads = nn.ModuleList()
        for i in range(num_semantic_classes):
            if self.virtual_depth_variation == 0:  # Propabilities both scale and canonical
                self.p_heads.append(ProbCanonicalHead(in_dim, hidden_dim, bin_num=bin_num))
                self.s_heads.append(ProbScaleHead(in_dim, num_bins=self.bins_scale))
            elif self.virtual_depth_variation == 1:  # Probabilities canonical
                self.p_heads.append(ProbCanonicalHead(in_dim, hidden_dim, bin_num=bin_num))
                self.s_heads.append(RegressScaleHead(in_dim))
            elif self.virtual_depth_variation == 2:  # Probabilities scale
                self.p_heads.append(RegressCanonicalHead(in_dim, hidden_dim))
                self.s_heads.append(ProbScaleHead(in_dim, num_bins=self.bins_scale))
            else:  # Regression
                self.p_heads.append(RegressCanonicalHead(in_dim, hidden_dim))
                self.s_heads.append(RegressScaleHead(in_dim))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, masks, min_scale=0,
                max_scale=15, max_tree_depth=1):

        # Depth #
        pred_depths_r_list = []  # Metric
        pred_depths_rc_list = []  # Canonical
        pred_depths_c_list = []  # Labels

        # Scale #
        pred_scale_list = []
        pred_depths_scale_c_list = []

        # Uncertainty #
        uncertainty_maps_list = []
        uncertainty_maps_scale_list = []

        # Canonical bins #
        if self.virtual_depth_variation == 0 or self.virtual_depth_variation == 1:
            bins_map, bin_edges = get_uniform_bins(depth, min_depth, max_depth, bin_num, sid=self.sid_canonical)

        # Scale bins #
        if self.virtual_depth_variation == 0 or self.virtual_depth_variation == 2:
            bins_map_scale, bin_scale_edges = get_uniform_bins(
                torch.zeros(depth.shape[0], 1, 1, 1).to(depth.device), min_scale, max_scale, self.bins_scale,
                sid=self.sid_scale)

        b, _, h, w = depth.shape
        h *= 4
        w *= 4
        # Initialize to zeros in case of regression #
        uncertainty_map = torch.zeros(b, self.num_semantic_classes, h, w).to(masks.device).clamp(min=1e-4)
        depth_c = torch.zeros(b, self.num_semantic_classes, h, w).to(masks.device)
        uncertainty_map_scale = torch.zeros(b, self.num_semantic_classes, 1, 1).to(masks.device).clamp(min=1e-4)
        depth_scale_c = torch.zeros(b, self.num_semantic_classes, 1, 1).to(masks.device)

        # Keep predictions per semantic class #
        depth_rc = []
        pred_scale = []
        for i in range(self.num_semantic_classes):

            # Concat mask of current class #
            if self.concat_masks:
                input_feature_map_current = torch.cat((input_feature_map, masks[:, i, :, :].unsqueeze(1)), dim=1)
            else:
                input_feature_map_current = input_feature_map

            # Canonical: either values or probs #
            pred_rc = self.p_heads[i](input_feature_map_current)
            if self.virtual_depth_variation == 0 or self.virtual_depth_variation == 1:
                # Canonical depth estimation #
                depth_rc_current = (pred_rc * bins_map.detach()).sum(1, keepdim=True)

                # Uncertainty #
                uncertainty_map_current = torch.sqrt(
                    (pred_rc * ((bins_map.detach() - depth_rc_current.repeat(1, bin_num, 1, 1))**2)).sum(1, keepdim=True))

                # Labels #
                pred_label = get_label(torch.squeeze(depth_rc_current, 1), bin_edges, bin_num).unsqueeze(1)
                depth_c_current = torch.gather(bins_map.detach(), 1, pred_label.detach())

                # Upsample #
                depth_rc_current = upsample(depth_rc_current, scale_factor=4, upsample_type=self.upsample_type)
                depth_c_current = upsample(depth_c_current, scale_factor=4, upsample_type=self.upsample_type)
                uncertainty_map_current = upsample(uncertainty_map_current, scale_factor=4,
                                                   upsample_type=self.upsample_type, uncertainty=True)

                depth_rc.append(depth_rc_current)
                depth_c[:, i, :, :] = depth_c_current.squeeze(1)
                uncertainty_map[:, i, :, :] = uncertainty_map_current.squeeze(1)
            else:
                pred_rc = upsample(pred_rc, scale_factor=4, upsample_type=self.upsample_type)
                depth_rc_current = pred_rc
                depth_rc.append(depth_rc_current)

            # Scale: either values or probs #
            scale_current = self.s_heads[i](input_feature_map_current)
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
                scale_c_current = torch.gather(bins_map_scale.detach(), 1, pred_label.detach())
                depth_scale_c[:, i, :, :] = scale_c_current.squeeze(1)
            else:
                pred_scale.append(scale_current)


        # Save #
        depth_rc = torch.cat(depth_rc, dim=1)
        pred_depths_rc_list.append(depth_rc)
        
        pred_depths_c_list.append(depth_c)

        uncertainty_maps_list.append(uncertainty_map)
        
        pred_scale = torch.cat(pred_scale, dim=1)
        pred_scale_list.append(pred_scale)
        uncertainty_maps_scale_list.append(uncertainty_map_scale)
        pred_depths_scale_c_list.append(depth_scale_c)

        # Metric depth #
        if self.loss_type == 0:
            depth_r = (self.relu(depth_rc * pred_scale_list[-1].unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-4)
        else:
            depth_r = depth_rc * pred_scale_list[-1].unsqueeze(-1).unsqueeze(-1)

        pred_depths_r_list.append(depth_r)

        result = {}
        result["pred_depths_r_list"] = pred_depths_r_list
        result["pred_depths_rc_list"] = pred_depths_rc_list
        result["pred_depths_c_list"] = pred_depths_c_list

        result["pred_scale_list"] = pred_scale_list
        result["pred_depths_scale_c_list"] = pred_depths_scale_c_list

        result["uncertainty_maps_list"] = uncertainty_maps_list
        result["uncertainty_maps_scale_list"] = uncertainty_maps_scale_list
        
        return result


"""
Per-Instance scale
"""


class PerInstanceScale(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192, bin_num=16, loss_type=0, num_semantic_classes=14, \
                 num_instances=63, padding_instances=0, roi_align=False, roi_align_size=32, bins_scale=150, \
                 virtual_depth_variation=0, upsample_type=1, bins_type=0, bins_type_scale=0):
        super(PerInstanceScale, self).__init__()
        self.hidden_dim = hidden_dim
        self.loss_type = loss_type

        self.num_semantic_classes = num_semantic_classes
        self.num_instances = num_instances

        self.virtual_depth_variation = virtual_depth_variation
        self.upsample_type = upsample_type
        
        self.bins_type = bins_type
        self.bins_type_scale = bins_type_scale
        self.bins_scale = bins_scale

        self.sid_canonical = False if self.bins_type == 0 else True
        self.sid_scale = False if self.bins_type_scale == 0 else True

        self.roi_align = roi_align
        self.padding_instances = padding_instances


        if self.roi_align:
            output_size = (roi_align_size, roi_align_size)
            spatial_scale = 1.0 / 4
            sampling_ratio = 4

            self.roiAlign = RoIAlign(output_size, spatial_scale=spatial_scale, sampling_ratio=sampling_ratio)
            print("PerInstanceScale: using RoIAlign with sampling ratio: " + str(sampling_ratio) +
                  " and spatial scale: " + str(spatial_scale))
        else:
            print("PerInstanceScale: padding for bbox (RoISelect) set to: " + str(self.padding_instances))
            print("PerInstanceScale: using RoISelect")

        # Probabilities canonical/scale #
        if self.virtual_depth_variation == 0:  
            self.instances_canonical = ProbSharedCanonicalInstancesHead(self.hidden_dim, bin_num=bin_num)
            self.instances_scale_and_shift = ProbScaleInstancesHead(self.hidden_dim, downsampling=4,
                                                                    num_semantic_classes=self.num_semantic_classes,
                                                                    num_bins=self.bins_scale, sid=self.sid_scale)
            print("PerInstanceScale: bins for both scale and shift")
        # Propabilities canonical #
        elif self.virtual_depth_variation == 1:  
            self.instances_canonical = ProbSharedCanonicalInstancesHead(self.hidden_dim, bin_num=bin_num)
            self.instances_scale_and_shift = RegressScaleInstancesHead(self.hidden_dim, downsampling=4,
                                                                       num_semantic_classes=self.num_semantic_classes)
            print("PerInstanceScale: bins for canonical")
        # Propabilities scale #
        elif self.virtual_depth_variation == 2:  
            self.instances_canonical = RegressSharedCanonicalInstancesHead(self.hidden_dim, self.hidden_dim)
            self.instances_scale_and_shift = ProbScaleInstancesHead(self.hidden_dim, downsampling=4,
                                                                    num_semantic_classes=self.num_semantic_classes,
                                                                    num_bins=self.bins_scale, sid=self.sid_scale)
            print("PerInstanceScale: bins for scale")
        # Regression #
        else:  
            self.instances_canonical = RegressSharedCanonicalInstancesHead(self.hidden_dim, self.hidden_dim)
            self.instances_scale_and_shift = RegressScaleInstancesHead(self.hidden_dim, downsampling=4,
                                                                       num_semantic_classes=self.num_semantic_classes)
            print("PerInstanceScale: regression")

        self.relu = nn.ReLU(inplace=True)

    def forward(self, depth, context, input_feature_map, bin_num, min_depth, max_depth, masks, instances, boxes,
                labels):

        # Depth #
        pred_depths_r_list = []  # Metric
        pred_depths_rc_list = [] # Canonical
        pred_depths_c_list = []  # Labels

        # Scale #
        pred_scale_list = []
        pred_depths_scale_c_list = []

        # Uncertainty #
        uncertainty_maps_list = []
        uncertainty_maps_scale_list = []

        batch_size, _, h, w = depth.shape
        max_instances_size = instances.shape[1]

        # Mask feature maps based on bboxes #
        if self.roi_align:
            boxes_roi, _ = get_valid_boxes(boxes, labels)
            instances_per_batch, num_valid_boxes = get_valid_num_instances_per_batch(boxes, labels)
            instances_per_batch = torch.cat((torch.tensor([0]).to(labels.device), instances_per_batch)) 
            instances_per_batch_sum = torch.cumsum(instances_per_batch, dim=0)

            boxes_roi = boxes_roi.to(torch.float32)
            boxes_roi -= 0.5
            # List[Tensor[L, 4]]: size of list == batch_size                      #
            # L is different per batch, i.e. each batch has different valid boxes #
            boxes_roi = [(boxes_roi[instances_per_batch_sum[i]:instances_per_batch_sum[i + 1]])
                         for i in range(instances_per_batch_sum.shape[0] - 1)]
            
            # [num_valid_instances, channels, roi_align_size, roi_align_size] #
            input_feature_map_instances_roi = self.roiAlign(input_feature_map, boxes_roi)
        else:
            # [num_valid_instances, channels, h/4, w/4] #
            input_feature_map_instances_roi = roi_select_features(input_feature_map, boxes, labels,
                                                                  padding=self.padding_instances)

        # Canonical via bins #
        if self.virtual_depth_variation == 0 or self.virtual_depth_variation == 1:
            # [num_valid_instances, num_bins, h/4, w/4] # 
            instances_canonical_prob = self.instances_canonical(input_feature_map, boxes, labels)
            
            bins_map, bin_edges = get_uniform_bins(depth, min_depth, max_depth, bin_num, sid=self.sid_canonical)
            bins_map = bins_map[0, :, :, :].unsqueeze(0)
            bin_edges = bin_edges[0, :, :, :].unsqueeze(0)
            # [1, num_bins, h/4, w/4] # 
            
            # [num_valid_instances, num_bins, h/4, w/4] # 
            bins_map = torch.cat([bins_map] * instances_canonical_prob.shape[0], dim=0)
            bin_edges = torch.cat([bin_edges] * instances_canonical_prob.shape[0], dim=0)
            depth_rc = (instances_canonical_prob * bins_map.detach()).sum(1, keepdim=True)
            
            # Uncertainty #
            uncertainty_map = torch.sqrt(
                (instances_canonical_prob * ((bins_map.detach() - depth_rc.repeat(1, bin_num, 1, 1))**2)).sum(
                    1, keepdim=True))

            # Labels #
            pred_label = get_label(torch.squeeze(depth_rc, 1), bin_edges, bin_num).unsqueeze(1)
            depth_c = (torch.gather(bins_map.detach(), 1, pred_label.detach()))

            #######################
            # Fill full instances #
            #######################
            canonical_full = torch.zeros((batch_size * max_instances_size, 1, h, w)).to(depth_rc.device).clamp(min=1e-4)
            uncertainty_maps_full = torch.zeros(
                (batch_size * max_instances_size, 1, h, w)).to(depth_rc.device).clamp(min=1e-4)
            depth_c_full = torch.zeros((batch_size * max_instances_size, 1, h, w)).to(depth_rc.device).clamp(min=1e-4)

            # Find valid positions #
            valid_boxes = labels.view(batch_size * max_instances_size, 1)
            valid_boxes = torch.nonzero(valid_boxes != -1)

            # Fill canonical #
            canonical_full[valid_boxes[:, 0]] = depth_rc
            depth_rc = canonical_full.view(batch_size, max_instances_size, h, w)

            # Fill uncertainty #
            uncertainty_maps_full[valid_boxes[:, 0]] = uncertainty_map
            uncertainty_map = uncertainty_maps_full.view(batch_size, max_instances_size, h, w)

            # Fill labels #
            depth_c_full[valid_boxes[:, 0]] = depth_c
            depth_c = depth_c_full.view(batch_size, max_instances_size, h, w)

            # Upsample #
            depth_rc = upsample(depth_rc, scale_factor=4, upsample_type=self.upsample_type)
            uncertainty_map = upsample(uncertainty_map, scale_factor=4, upsample_type=self.upsample_type,
                                       uncertainty=True)
            depth_c = upsample(depth_c, scale_factor=4, upsample_type=self.upsample_type)

            pred_depths_rc_list.append(depth_rc)
            uncertainty_maps_list.append(uncertainty_map)
            pred_depths_c_list.append(depth_c)
        # Canonical regression #
        else:
            instances_canonical = self.instances_canonical(input_feature_map, boxes, labels)
            boxes_valid_idx = get_valid_boxes_idx(boxes, labels)

            # Fill full prediction with valid predictions #
            canonical_full = torch.zeros(
                (batch_size * self.num_instances, 1, h, w)).to(instances_canonical.device).clamp(min=1e-4)
            
            canonical_full[boxes_valid_idx[:, 0]] = instances_canonical
            canonical_full = canonical_full.view(batch_size, self.num_instances, h, w)

            # Updample #
            depth_rc = canonical_full
            depth_rc = upsample(depth_rc, scale_factor=4, upsample_type=self.upsample_type)

            pred_depths_rc_list.append(depth_rc)
            uncertainty_maps_list.append(torch.zeros_like(depth_rc).clamp(min=1e-4))
            pred_depths_c_list.append(torch.zeros_like(depth_rc))

        # Scale bins #
        if self.virtual_depth_variation == 0 or self.virtual_depth_variation == 2:
            # Predictions are done internally with ModuleList per class #
            # [num_valid_instances, num_semantic_classes]     
            instances_scale, unc_scale, scale_labels_bins = self.instances_scale_and_shift(
                input_feature_map_instances_roi, boxes, labels)
           
            # Pick correct prediction according to the class of the instance # 
            instances_scale = mask_predictions_to_true_class(instances_scale, labels)
            unc_scale = mask_predictions_to_true_class(unc_scale, labels)
            scale_labels_bins = mask_predictions_to_true_class(scale_labels_bins, labels)

            pred_scale_list.append(instances_scale)
            uncertainty_maps_scale_list.append(unc_scale)
            pred_depths_scale_c_list.append(scale_labels_bins)
        # Scale regression #
        else:
            instances_scale = self.instances_scale_and_shift(input_feature_map_instances_roi, boxes, labels)
            
            # Pick correct prediction according to the class of the instance # 
            instances_scale = mask_predictions_to_true_class(instances_scale, labels)
            
            pred_scale_list.append(instances_scale)
            uncertainty_maps_scale_list.append(torch.zeros_like(instances_scale).clamp(min=1e-4))
            pred_depths_scale_c_list.append(torch.zeros_like(instances_scale))

        # Metric
        if self.loss_type == 0:
            depth_instances_r = (self.relu(depth_rc * instances_scale.unsqueeze(-1).unsqueeze(-1))).clamp(min=1e-4)
        else:
            depth_instances_r = depth_rc * instances_scale.unsqueeze(-1).unsqueeze(-1)

        # Size: b, 1, h, w #
        pred_depths_r_list.append(depth_instances_r)
        result = {}

        result["pred_depths_r_list"] = pred_depths_r_list
        result["pred_depths_rc_list"] = pred_depths_rc_list
        result["pred_depths_c_list"] = pred_depths_c_list

        result["uncertainty_maps_list"] = uncertainty_maps_list
        result["uncertainty_maps_scale_list"] = uncertainty_maps_scale_list

        result["pred_scale_list"] = pred_scale_list
        result["pred_depths_scale_c_list"] = pred_depths_scale_c_list

        return result
