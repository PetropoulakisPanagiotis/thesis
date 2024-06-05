import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2

from .depth_update_clean import padding_global


"""
ProjectionInputDepth: project bin depth canditates (x4 cnv)
"""
class ProjectionInputDepth(nn.Module):
    def __init__(self, hidden_dim, out_chs, bin_num):
        super().__init__()
        self.out_chs = out_chs 
        self.convd1 = nn.Conv2d(bin_num, hidden_dim, 7, padding=3)
        self.convd2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.convd3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.convd4 = nn.Conv2d(hidden_dim, out_chs, 3, padding=1)
        
    def forward(self, depth):
        d = F.relu(self.convd1(depth))
        d = F.relu(self.convd2(d))
        d = F.relu(self.convd3(d))
        d = F.relu(self.convd4(d))
                
        return d


"""
Projection: same convolution (x1 cnv)
"""
class Projection(nn.Module):
    def __init__(self, in_chs, out_chs):
        super().__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, 3, padding=1)
        
    def forward(self, x):
        out = self.conv(x)
                
        return out


"""
ProjectionV2: same convolution (x2 cnv)
"""
class ProjectionV2(nn.Module):
    def __init__(self, in_chs, out_chs, hidden_dim=96):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chs, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, out_chs, 3, padding=1)
        
    def forward(self, x):
        out = self.conv2(F.relu(self.conv1(x)))
                
        return out


"""
Upsample input tensor by a factor of 2
"""
def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners, recompute_scale_factor=False)


def get_uniform_bins(feature_map, min_depth=0, max_depth=0, bin_num=5):
    with torch.no_grad():    
        b, _, h, w = feature_map.shape

        interval = (max_depth - min_depth) / bin_num
        interval = torch.ones(b, bin_num + 1, h, w, device=feature_map.device) * interval
        
        bins = torch.cumsum(interval, 1).clamp(min_depth, max_depth)

    return bins


def get_iebins(feature_map, min_depth=0, max_depth=0, bin_num=5):
    with torch.no_grad():    
        depth_range = max_depth - min_depth
        interval = depth_range / bin_num

        interval = interval * torch.ones_like(feature_map)
        interval = interval.repeat(1, bin_num, 1, 1)
        interval = torch.cat([torch.ones_like(feature_map) * min_depth, interval], 1)

        bin_edges = torch.cumsum(interval, 1)
        bins = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
    
    return bin_edges, bins


def update_bins(bin_edges, target_bin_left, target_bin_right, depth_r, pred_label, bin_num, min_depth, max_depth, uncertainty_range):
    with torch.no_grad():    
        b, _, h, w = bin_edges.shape

        depth_range = uncertainty_range
        depth_start_update = torch.clamp_min(depth_r - 0.5 * depth_range, min_depth)

        interval = depth_range / bin_num
        interval = interval.repeat(1, bin_num, 1, 1)
        interval = torch.cat([torch.ones([b, 1, h, w], device=bin_edges.device) * depth_start_update, interval], 1)

        bin_edges = torch.cumsum(interval, 1).clamp(min_depth, max_depth)
        curr_depth = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        
    return bin_edges.detach(), curr_depth.detach()


"""
Get bin idx
"""
def get_label(depth_prediction, bin_edges, bin_num):
    with torch.no_grad():
        label_bin = torch.zeros(depth_prediction.size(), dtype=torch.int64, device=depth_prediction.device)

        for i in range(bin_num):
            bin_mask = torch.ge(depth_prediction, bin_edges[:, i])
            bin_mask = torch.logical_and(bin_mask, torch.lt(depth_prediction, bin_edges[:, i + 1]))
        
            label_bin[bin_mask] = i
        
        return label_bin


"""
Boxes operations start
"""


def normalize_box(box, height=480, width=640):
    with torch.no_grad():
        return torch.stack(((box[:, 0] / height).float(), 
                            (box[:, 1] / width).float(), 
	                        (box[:, 2] / height).float(), 
	                        (box[:, 3] / width).float()), dim=1)


def project_box_to_feature_map(box, downsampling, height=480, width=640, padding=0):
    global padding_global    
    padding = padding_global
    
    with torch.no_grad():
        if padding == 0:
            return torch.ceil(box / downsampling).int()
        else:
            new_box = torch.ceil(box / downsampling).int()
            height /= downsampling
            width /= downsampling

            new_box = torch.stack((
                (new_box[:, 0] - padding).clamp(min=0).int(), 
                (new_box[:, 1] - padding).clamp(min=0).int(), 
                (new_box[:, 2] + padding).clamp(max=height-1).int(), 
                (new_box[:, 3] + padding).clamp(max=width-1).int()), dim=1)

            return new_box 


def get_valid_boxes_idx(boxes, labels):
    with torch.no_grad():
        b, num_instances, _ = boxes.shape
        boxes_reshaped = boxes.view(b * num_instances, 4)

        labels_reshaped = labels.view(b * num_instances, 1)
        boxes_valid_idx = torch.nonzero(labels_reshaped != -1)

    return boxes_valid_idx


def get_valid_boxes(boxes, labels):
    with torch.no_grad():
        b, num_instances, _ = boxes.shape
        boxes_reshaped = boxes.view(b * num_instances, 4)

        labels_reshaped = labels.view(b * num_instances, 1)
        boxes_valid_idx = torch.nonzero(labels_reshaped != -1)

        boxes_valid = boxes_reshaped[boxes_valid_idx[:, 0]]
        num_valid_boxes, _ = boxes_valid.shape

    return boxes_valid, num_valid_boxes


def get_valid_normalized_projected_boxes(feature_map, boxes, labels, downsampling):
    h, w = feature_map.shape[2:]

    with torch.no_grad():
        boxes_valid, num_valid_boxes = get_valid_boxes(boxes, labels)
        boxes_valid_projected = project_box_to_feature_map(boxes_valid, downsampling)
        boxes_valid_normalized_projected = normalize_box(boxes_valid_projected, height=h, width=w)

    return boxes_valid_normalized_projected, num_valid_boxes


def get_valid_num_instances_per_batch(boxes, labels):
    with torch.no_grad():

        boxes_valid, num_valid_boxes = get_valid_boxes(boxes, labels)

        instances_per_batch = torch.nonzero(labels != -1)
        instances_per_batch = torch.bincount(instances_per_batch[:, 0])
    

    return instances_per_batch, num_valid_boxes


"""
Boxes operations end
"""


"""
Select features operations start
"""


def roi_select_features(feature_map, boxes, labels, downsampling=4):
    with torch.no_grad():
        height, width = feature_map.shape[2:] 

        boxes_valid, _ = get_valid_boxes(boxes, labels)
        instances_per_batch, num_valid_boxes = get_valid_num_instances_per_batch(boxes, labels)

        box_coordinates_projected = project_box_to_feature_map(boxes_valid, downsampling)

        ymin, xmin, ymax, xmax = box_coordinates_projected.split(1, dim=1)

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
                x = upsample(masked_feature_map[i, :, :, :].unsqueeze(1), 4)
                x = x[i, 0, :, :].unsqueeze(0).permute(1,2,0)
                x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
                x = (x.cpu().detach().numpy() * 255).astype('uint8')
                cv2.imshow(str(i), x)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        return masked_feature_map


# Output of this function: each instance feature map has a masked decoder feature map were all areas of the same class has been kept active
def roi_select_features_canonical_shared(feature_map, boxes, labels, downsampling=4, num_semantic_classes=14):
    with torch.no_grad():
        batch_size, num_max_instances = boxes.shape[0:2]
        height, width = feature_map.size(-2), feature_map.size(-1)

        feature_maps_final = torch.zeros((batch_size, num_semantic_classes-1, feature_map.shape[1], height, width)).to(labels.device)
        
        for class_i in range(num_semantic_classes - 1):
            instances_per_batch = torch.nonzero(labels == class_i)
            instances_per_batch = torch.bincount(instances_per_batch[:, 0])

            # Pick all boxes of the same class #
            valid_boxes = labels.view(batch_size * num_max_instances, 1)
            valid_boxes = torch.nonzero(valid_boxes == class_i)
            
            if valid_boxes.shape[0] == 0:
                masked_feature_map = torch.zeros_like(feature_map)
            else:        
                box_coordinates = boxes.view(batch_size * num_max_instances, 4)
                box_coordinates = box_coordinates[valid_boxes[:, 0]]
                
                box_coordinates = project_box_to_feature_map(box_coordinates, downsampling)

                ymin, xmin, ymax, xmax = box_coordinates.split(1, dim=1)

                row_indices = torch.arange(height, device=feature_map.device).unsqueeze(0)
                col_indices = torch.arange(width, device=feature_map.device).unsqueeze(0)

                row_mask = (row_indices >= ymin) & (row_indices <= ymax)
                col_mask = (col_indices >= xmin) & (col_indices <= xmax)

                row_mask = row_mask.unsqueeze(1).unsqueeze(-1)
                col_mask = col_mask.unsqueeze(1).unsqueeze(2)

                zeros_mask = torch.cat([torch.zeros_like(feature_map[i, :, :, :].unsqueeze(0)).repeat(times,1,1,1) for i, times in enumerate(instances_per_batch)], dim=0)

                instances_per_batch_non_zero_idx = instances_per_batch.nonzero()
                instances_per_batch = instances_per_batch[instances_per_batch_non_zero_idx].squeeze(-1)
                
                instances_per_batch = torch.cat((torch.tensor([1]).to(labels.device), instances_per_batch))
                instances_per_batch = torch.cumsum(instances_per_batch, dim=0)

                masks = (zeros_mask + row_mask) * (zeros_mask + col_mask)
                batches_masks = torch.cat([torch.sum(masks[instances_per_batch[i] - 1:instances_per_batch[i+1]], dim=0).unsqueeze(0) for i in range(instances_per_batch.shape[0] - 1)], dim=0)
                masked_feature_map = feature_map[instances_per_batch_non_zero_idx].squeeze(1) * batches_masks
                feature_maps_final[instances_per_batch_non_zero_idx[:], class_i, :, :, :] = masked_feature_map.unsqueeze(1)
                feature_maps_final = feature_maps_final.squeeze(1)

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


# Return masked feature maps of a specific class # 
def roi_select_features_module(feature_map, box, labels, class_label, downsampling=4):
    with torch.no_grad():
        batch_size, num_max_instances = box.shape[0:2]
        
        height, width = feature_map.size(-2), feature_map.size(-1)
        box_coordinates = box.view(batch_size * num_max_instances, 4)

        # Pick given class boxes #
        instances_per_batch = torch.nonzero(labels == class_label)
        instances_per_batch = torch.bincount(instances_per_batch[:, 0])
        
        labels_reshaped = labels.view(batch_size * num_max_instances, 1)
        valid_boxes = torch.nonzero(labels_reshaped == class_label)

        box_coordinates = box_coordinates[valid_boxes[:, 0]]
        box_coordinates = project_box_to_feature_map(box_coordinates, downsampling)
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


"""
Select features operations end
"""


"""
Select out operations start
"""


def pick_predictions_instances_scale_shift(prediction_scale_shift, labels, only_scale=False):

    batch_size, labels_max_size = labels.shape[0:2]

    with torch.no_grad():
        labels_reshaped = labels.view(batch_size*labels_max_size)
        labels_valid_idx = torch.nonzero(labels_reshaped != -1)
        labels_valid_num = labels_valid_idx.shape[0]

        labels_valid = labels_reshaped[labels_valid_idx[:,0]]

    if only_scale:
        pred_scale = prediction_scale_shift
    
        # Pick scale/shift that corresponds to the correct class #
        pred_scale = pred_scale[torch.arange(labels_valid_num), labels_valid].unsqueeze(-1)

        pred_scale_full = torch.zeros((batch_size*labels_max_size, 1)).to(prediction_scale_shift.device)

        pred_scale_full[labels_valid_idx[:,0]] = pred_scale

        pred_scale_full = pred_scale_full.view(batch_size, labels_max_size)

        return pred_scale_full, None

    else:
        pred_scale = prediction_scale_shift[:, ::2]
        pred_shift = prediction_scale_shift[:, 1::2]
    
        # Pick scale/shift that corresponds to the correct class #
        pred_scale = pred_scale[torch.arange(labels_valid_num), labels_valid].unsqueeze(-1)
        pred_shift = pred_shift[torch.arange(labels_valid_num), labels_valid].unsqueeze(-1)

        pred_scale_full = torch.zeros((batch_size*labels_max_size, 1)).to(prediction_scale_shift.device)
        pred_shift_full = torch.zeros((batch_size*labels_max_size, 1)).to(prediction_scale_shift.device)

        pred_scale_full[labels_valid_idx[:,0]] = pred_scale
        pred_shift_full[labels_valid_idx[:,0]] = pred_shift

        pred_scale_full = pred_scale_full.view(batch_size, labels_max_size)
        pred_shift_full = pred_shift_full.view(batch_size, labels_max_size)

        return pred_scale_full, pred_shift_full
    

"""
Only scale
"""
def pick_predictions_instances_scale(pred_scale, labels):

    batch_size, labels_max_size = labels.shape[0:2]

    with torch.no_grad():
        labels_reshaped = labels.view(batch_size*labels_max_size)
        labels_valid_idx = torch.nonzero(labels_reshaped != 0)
        labels_valid_num = labels_valid_idx.shape[0]

        labels_valid = labels_reshaped[labels_valid_idx[:,0]]

    # Pick scale/shift that corresponds to the correct class #
    pred_scale = pred_scale[torch.arange(labels_valid_num), labels_valid].unsqueeze(-1)

    pred_scale_full = torch.zeros((batch_size*labels_max_size, 1)).to(pred_scale.device)

    pred_scale_full[labels_valid_idx[:,0]] = pred_scale

    pred_scale_full = pred_scale_full.view(batch_size, labels_max_size)

    return pred_scale_full

def pick_predictions_instances_canonical(prediction, labels): 
    batch_size, labels_max_size = labels.shape[0:2]
    h, w = prediction.shape[2:]

    # Prediction --> num_classes - 1 does not include null #
    with torch.no_grad():
        labels_reshaped = labels.view(batch_size*labels_max_size)
        labels_valid_idx = torch.nonzero(labels_reshaped != 0)
        labels_valid_num = labels_valid_idx.shape[0]

        labels_valid = labels_reshaped[labels_valid_idx[:,0]]

    canonical = prediction[torch.arange(labels_valid_num), labels_valid].unsqueeze(1)

    canonical_full = torch.zeros((batch_size*labels_max_size, 1, h, w)).to(prediction.device)
    canonical_full[labels_valid_idx[:,0]] = canonical 

    canonical_full = canonical_full.view(batch_size, labels_max_size, h, w)
    
    return canonical_full


def pick_predictions_instances_canonical_shared_class(prediction, labels):
    batch_size, num_max_instances = labels.shape[0:2]
    valid_instances, num_semantic_classes, h, w = prediction.shape # -1 already 
    
    # Each batch has maps == num_semantic_classes and for each we predict num_semantic_classes predictions 
    prediction = prediction.view(batch_size*num_semantic_classes*num_semantic_classes, h, w)   

    # Pick the correct prediction for each class map #
    prediction = prediction[torch.tensor(range(0, batch_size*num_semantic_classes*num_semantic_classes, num_semantic_classes)).to(labels.device), :, :]
    prediction = prediction.view(batch_size, num_semantic_classes, h, w) 
    
    prediction = prediction.unsqueeze(1).repeat(1, num_max_instances, 1,1,1)
    prediction = prediction.view(batch_size*num_max_instances, num_semantic_classes, h, w)

    # Pick the correct canonical class for the instances #
    with torch.no_grad():
        labels_fast = torch.where(labels == -1, 1, labels) 
        labels_fast = labels_fast.view(batch_size*num_max_instances)
    
    prediction = prediction[torch.arange(batch_size*num_max_instances), labels_fast]
    prediction = prediction.view(batch_size, num_max_instances, h, w)

    return prediction


"""
Select out operations end
"""
