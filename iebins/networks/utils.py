import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
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


def bilinear_weights(height, width, scale_factor, device):
    with torch.no_grad():
        grid_y, grid_x = torch.meshgrid(torch.arange(0, height * scale_factor) / scale_factor, \
                                        torch.arange(0, width * scale_factor) / scale_factor)
        grid_x = grid_x.to(device)
        grid_y = grid_y.to(device)

        x1 = torch.floor(grid_x).long()
        y1 = torch.floor(grid_y).long()
        x2 = torch.ceil(grid_x).long()
        y2 = torch.ceil(grid_y).long()

        x1 = torch.clamp(x1, 0, width - 2)
        y1 = torch.clamp(y1, 0, height - 2)
        x2 = torch.clamp(x2, 1, width - 1)
        y2 = torch.clamp(y2, 1, height - 1)

        # Compute the distances
        dx = grid_x - x1.float()
        dy = grid_y - y1.float()

        # Compute the complementary distances
        dx1 = 1 - dx
        dy1 = 1 - dy

        # Compute the weights
        w00 = dx1 * dy1
        w01 = dx * dy1
        w10 = dx1 * dy
        w11 = dx * dy

    return w00, w01, w10, w11, x1, y1, x2, y2


def upsample_custom(x, scale_factor=2, uncertainty=False):
    _, _, height, width = x.shape
    w00, w01, w10, w11, x1, y1, x2, y2 = bilinear_weights(height, width, scale_factor, device=x.device)

    d00 = x[:, :, y1, x1]
    d01 = x[:, :, y2, x1]
    d10 = x[:, :, y1, x2]
    d11 = x[:, :, y2, x2]
    if uncertainty:
        depth_upsampled = w00**2 * d00 + w01**2 * d01 + w10**2 * d10 + w11**2 * d11
    else:
        depth_upsampled = w00 * d00 + w01 * d01 + w10 * d10 + w11 * d11

    return depth_upsampled


"""
Upsample input tensor
"""


def upsample(x, scale_factor=2, mode="bilinear", align_corners=False, upsample_type=1, uncertainty=False):
    if upsample_type == 0:
        return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners,
                             recompute_scale_factor=False)
    elif upsample_type == 1:
        return upsample_custom(x, scale_factor, False)
    else:
        return upsample_custom(x, scale_factor, True) if uncertainty else upsample_custom(x, scale_factor, False)


"""
sid: use log space Fu et. al. 2018 
"""


def get_uniform_bins(feature_map, min_depth=0, max_depth=0, bin_num=5, sid=True):
    with torch.no_grad():
        b, _, h, w = feature_map.shape
        if sid:
            xi = torch.tensor(1 - min_depth).to(feature_map.device)
            beta_star = max_depth + xi

            indices = torch.arange(bin_num + 1, dtype=torch.float32).to(feature_map.device)

            bin_edges = torch.exp(torch.log(beta_star) * indices / bin_num) - xi
            bin_edges = bin_edges.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(b, -1, h, w)
            bins = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        else:
            interval = (max_depth - min_depth) / bin_num

            interval = torch.ones(b, bin_num, h, w, device=feature_map.device) * interval
            interval = torch.cat([torch.ones_like(feature_map) * min_depth, interval], 1)

            bin_edges = torch.cumsum(interval, 1).clamp(min_depth, max_depth)
            bins = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])

    return bins.detach(), bin_edges.detach()


def get_iebins(feature_map, min_depth=0, max_depth=0, bin_num=5):
    with torch.no_grad():
        depth_range = max_depth - min_depth
        interval = depth_range / bin_num

        interval = interval * torch.ones_like(feature_map)
        interval = interval.repeat(1, bin_num, 1, 1)
        interval = torch.cat([torch.ones_like(feature_map) * min_depth, interval], 1)

        bin_edges = torch.cumsum(interval, 1)
        bins = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])

    return bins, bin_edges


"""
Update bin canditates based on the target bin and predicted variance
"""


def update_bins(bin_edges, target_bin_left, target_bin_right, depth_r, pred_label, bin_num, min_depth, max_depth,
                uncertainty_range):
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
    # xmin, ymin, xmax, ymax (w, h, w, h)
    with torch.no_grad():
        return torch.stack(((box[:, 0] / width).float(), (box[:, 1] / height).float(), (box[:, 2] / width).float(),
                            (box[:, 3] / height).float()), dim=1)


"""
h, w should be divisible by downsampling
"""


def project_box_to_feature_map(box, downsampling, height=480, width=640, padding=0):
    with torch.no_grad():
        if padding == 0:
            return torch.ceil(box / downsampling).int()
        else:
            new_box = torch.ceil(box / downsampling).int()
            height /= downsampling
            width /= downsampling

            new_box = torch.stack(
                ((new_box[:, 0] - padding).clamp(min=0).int(), (new_box[:, 1] - padding).clamp(min=0).int(),
                 (new_box[:, 2] + padding).clamp(max=width - 1).int(),
                 (new_box[:, 3] + padding).clamp(max=height - 1).int()), dim=1)

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


def get_valid_normalized_projected_boxes(feature_map, boxes, labels, downsampling, padding=0):
    h, w = feature_map.shape[2:]

    with torch.no_grad():
        boxes_valid, num_valid_boxes = get_valid_boxes(boxes, labels)
        boxes_valid_projected = project_box_to_feature_map(boxes_valid, downsampling, padding)
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
Maskout feature that do not belong to the given box
This is done for every instance
"""


def roi_select_features(feature_map, boxes, labels, downsampling=4, padding=0):
    """
        feature_map: [batch_size, c, h, w]
        masked_feature_map: [num_valid_instances, c, h, w]
    """
    with torch.no_grad():
        height, width = feature_map.shape[2:]

        boxes_valid, _ = get_valid_boxes(boxes, labels)
        instances_per_batch, num_valid_boxes = get_valid_num_instances_per_batch(boxes, labels)

        box_coordinates_projected = project_box_to_feature_map(boxes_valid, downsampling, padding)

        ymin, xmin, ymax, xmax = box_coordinates_projected.split(1, dim=1)

        row_indices = torch.arange(height, device=feature_map.device).unsqueeze(0)
        col_indices = torch.arange(width, device=feature_map.device).unsqueeze(0)

        row_mask = (row_indices >= ymin) & (row_indices <= ymax)
        col_mask = (col_indices >= xmin) & (col_indices <= xmax)

        row_mask = row_mask.unsqueeze(1).unsqueeze(-1)
        col_mask = col_mask.unsqueeze(1).unsqueeze(2)
        zeros_mask = torch.cat([
            torch.zeros_like(feature_map[i, :, :, :].unsqueeze(0)).repeat(times, 1, 1, 1)
            for i, times in enumerate(instances_per_batch)
        ], dim=0)
        masks = (zeros_mask + row_mask) * (zeros_mask + col_mask)

        masked_feature_map = torch.cat([
            feature_map[i, :, :, :].unsqueeze(0).repeat(times, 1, 1, 1) for i, times in enumerate(instances_per_batch)
        ], dim=0) * masks

        if False:  # Debug viz
            for i in range(masked_feature_map.shape[0]):
                x = upsample(masked_feature_map[i, :, :, :].unsqueeze(1), 4)
                x = x[i, 0, :, :].unsqueeze(0).permute(1, 2, 0)
                x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
                x = (x.cpu().detach().numpy() * 255).astype('uint8')
                cv2.imshow(str(i), x)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        return masked_feature_map


"""
In: [num_valid_instances, num_classes]
Out: [batch_size, max_instances]
"""


def mask_predictions_to_true_class(prediction_estimate, labels):
    # Get valid labels #
    with torch.no_grad():
        batch_size, labels_max_size = labels.shape[0:2]
        labels_reshaped = labels.view(batch_size * labels_max_size)
        labels_valid_idx = torch.nonzero(labels_reshaped != -1)

        labels_valid_num = labels_valid_idx.shape[0]
        labels_valid = labels_reshaped[labels_valid_idx[:, 0]]

    # Pick prediction that corresponds to the correct class #
    prediction_estimate = prediction_estimate[torch.arange(labels_valid_num), labels_valid].unsqueeze(-1)

    # For empty instances fill with zero #
    prediction_estimate_full = torch.zeros(
        (batch_size * labels_max_size, 1)).to(prediction_estimate.device).clamp(min=1e-3)
    prediction_estimate_full[labels_valid_idx[:, 0]] = prediction_estimate
    prediction_estimate_full = prediction_estimate_full.view(batch_size, labels_max_size)

    return prediction_estimate_full
