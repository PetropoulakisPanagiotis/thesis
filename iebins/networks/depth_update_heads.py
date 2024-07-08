import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *
"""
Prediction heads for metric, canonical and scale 
"""
"""
ProbCanonicalHead: propabilities head for bin prediction
"""


class ProbCanonicalHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, bin_num=16):
        super(ProbCanonicalHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, bin_num, 3, padding=1)

    def forward(self, x):
        out = torch.softmax(self.conv2(F.relu(self.conv1(x))), 1)
        return out


"""
RegressCanonicalHead: regression head for canonical prediction [0,1]
"""


class RegressCanonicalHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super(RegressCanonicalHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)

    def forward(self, x):
        out = torch.sigmoid(self.conv2(F.relu(self.conv1(x))))
        return out


"""
RegressScaleHead: scale regression head
Out: 1-dim scale
"""


class RegressScaleHead(nn.Module):
    def __init__(self, input_dim=128):
        super(RegressScaleHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(64)
        self.fc1 = nn.Linear(64 * 64, 1)

    def forward(self, x):
        out = F.relu(self.pool(self.conv1(x)))
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out)).clamp(min=1e-4)
        return out


"""
ProbScaleHead: scale head bins
Out: 1-dim scale
Advice: Use ModuleList
"""


class ProbScaleHead(nn.Module):
    def __init__(self, input_dim=128, num_bins=50):
        super(ProbScaleHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(64)
        self.fc1 = nn.Linear(64 * 64, num_bins)

    def forward(self, x):
        out = F.relu(self.pool(self.conv1(x)))
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = torch.softmax(out, axis=1)
        return out


"""
Scale prediction per instance via bins
Also concat the bbox into the features
Out: [num_valid_instances, num_semantic_classes]
"""


class ProbScaleInstancesHead(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=13, num_bins=50, min_scale=0, max_scale=15,
                 sid=False):
        super(ProbScaleInstancesHead, self).__init__()
        self.downsampling = downsampling
        self.num_bins = num_bins
        self.input_dim = input_dim
        self.num_semantic_classes = num_semantic_classes
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.sid = sid

        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(64)

        # One head per class #
        self.scale_nets = nn.ModuleList()
        for i in range(num_semantic_classes):
            self.scale_nets.append(nn.Linear((64 * 64) + 4, self.num_bins))

    def forward(self, x, boxes, labels):
        """
            x: [num_valid_instances, channels, h, w] #
        """
        bins_map_scale, bin_edges_scale = get_uniform_bins(torch.zeros(x.shape[0], 1, 1, 1).to(x.device), \
                                                           self.min_scale, self.max_scale, self.num_bins, self.sid)
        bins_map_scale = bins_map_scale.squeeze(-1).squeeze(-1)
        bin_edges_scale = bin_edges_scale.squeeze(-1).squeeze(-1)

        boxes_valid_normalized_projected, num_valid_boxes = get_valid_normalized_projected_boxes(
            x, boxes, labels, self.downsampling)

        out = self.pool(F.relu(self.conv1(x)))
        out = torch.flatten(out, 1)
        out = torch.cat((out, boxes_valid_normalized_projected), dim=1)  # Concat boxes

        out_list_scale, out_list_unc, out_list_labels = [], [], []
        for i in range(self.num_semantic_classes):

            probs = torch.softmax(self.scale_nets[i](out), axis=1)
            scale = (probs * bins_map_scale.detach()).sum(1, keepdim=True)

            uncertainty = torch.sqrt((probs * ((bins_map_scale.detach() - scale)**2)).sum(1, keepdim=True))

            pred_label = get_label(torch.squeeze(scale, 1), bin_edges_scale, self.num_bins).unsqueeze(1)
            depth_c = (torch.gather(bins_map_scale.detach(), 1, pred_label.detach()))

            out_list_scale.append(scale)
            out_list_unc.append(uncertainty)
            out_list_labels.append(depth_c)

        out_scale = torch.cat(out_list_scale, dim=1)
        out_unc = torch.cat(out_list_unc, dim=1)
        out_labels = torch.cat(out_list_labels, dim=1)

        return out_scale, out_unc, out_labels


"""
Predict scales per instance via regression sigmoid 
Out: [num_valid_instances, num_semantic_classes]
"""


class RegressScaleInstancesHead(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(RegressScaleInstancesHead, self).__init__()
        self.downsampling = downsampling
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(64)

        # One head per class #
        self.scale_nets = nn.ModuleList()
        for i in range(num_semantic_classes):
            self.scale_nets.append(nn.Linear((64 * 64) + 4, 1))

    def forward(self, x, boxes, labels):
        """
            x: [num_valid_instances, channels, h, w] #
        """
        boxes_valid_normalized_projected, num_valid_boxes = get_valid_normalized_projected_boxes(
            x, boxes, labels, self.downsampling)

        out = self.pool(F.relu(self.conv1(x)))
        out = torch.flatten(out, 1)
        out = torch.cat((out, boxes_valid_normalized_projected), dim=1)

        out_list = []
        for i in range(self.num_semantic_classes):
            scale = F.relu(self.scale_nets[i](out)).clamp(min=1e-4)
            out_list.append(scale)

        out = torch.cat(out_list, dim=1)
        return out


"""
Shared canonical prediction for instances via bins
"""


class ProbSharedCanonicalInstancesHead(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=128, bin_num=40):
        super(ProbSharedCanonicalInstancesHead, self).__init__()

        self.input_dim = input_dim
        self.bin_num = bin_num

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, bin_num, 3, padding=1)

    def forward(self, x, boxes, labels):
        """
            x: [batch_size, channels, h, w]
        """
        h, w = x.shape[2:]

        instances_per_batch, num_valid_boxes = get_valid_num_instances_per_batch(boxes, labels)

        out = F.relu(self.conv1(x))
        out = torch.softmax(self.conv2(out), 1)

        # Duplicate canonical to instances #
        out = torch.cat(
            [out[i, :, :, :].unsqueeze(0).repeat(times, 1, 1, 1) for i, times in enumerate(instances_per_batch)], dim=0)
        out = out.view(num_valid_boxes, self.bin_num, h, w)

        return out


"""
Shared canonical prediction for instances via regression sigmoid
"""


class RegressSharedCanonicalInstancesHead(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=128):
        super(RegressSharedCanonicalInstancesHead, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)

    def forward(self, x, boxes, labels):

        h, w = x.shape[2:]

        instances_per_batch, num_valid_boxes = get_valid_num_instances_per_batch(boxes, labels)

        out = F.relu(self.conv1(x))
        out = torch.sigmoid(self.conv2(out))

        out = torch.cat(
            [out[i, :, :, :].unsqueeze(0).repeat(times, 1, 1, 1) for i, times in enumerate(instances_per_batch)], dim=0)
        out = out.view(num_valid_boxes, 1, h, w)

        return out


"""
Other
"""


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128 + 192):
        super(SepConvGRU, self).__init__()

        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))

        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h
