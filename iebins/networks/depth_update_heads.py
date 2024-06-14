import torch
import torch.nn as nn
import torch.nn.functional as F

from .depth_update import padding_global
from .utils import *


"""
Canonical/Depth and scale/shift heads
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
SSHead: scale and shift regression head - single per image 
"""     
class SSHead(nn.Module):
    def __init__(self, input_dim=128, num_out=1):
        super(SSHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(88)
        self.fc1 = nn.Linear(88 * 88, num_out) 
    
    def forward(self, x):
        out = F.relu(self.pool(self.conv1(x)))
        out = torch.flatten(out, 1)
        out = self.fc1(out) 
       
        return out

"""
SHead: scale head bins - single per image 
"""     
class SHead(nn.Module):
    def __init__(self, input_dim=128, num_bins=50):
        super(SHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(88)
        self.fc1 = nn.Linear(88 * 88, num_bins) 
    
    def forward(self, x):
        out = F.relu(self.pool(self.conv1(x)))
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = torch.softmax(out, axis=1)
       
        return out


""""
ROISelectScale start
"""

"""
Predict scale/shift for a ROI with bbox concat
"""
class ROISelectScale(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_out=14):
        super(ROISelectScale, self).__init__()
        self.downsampling = downsampling   
        self.num_out = num_out
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(70)
        self.fc1 = nn.Linear((70*70) + 4, num_out) # Scale and shift 
 
    def forward(self, x, boxes, labels):
        boxes_valid_normalized_projected, num_valid_boxes = get_valid_normalized_projected_boxes(x, boxes, labels, self.downsampling)
        out = self.pool(F.relu(self.conv1(x)))

        out = torch.flatten(out, 1)
        out = torch.cat((out, boxes_valid_normalized_projected), dim=1) # Concat boxes

        out = self.fc1(out)
        out = out.view(num_valid_boxes, self.num_out)
        
        return out
        

"""
Predict scale with bins for a ROI with bbox concat
"""
class ROISelectScaleBins(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=13, num_bins=50):
        super(ROISelectScaleBins, self).__init__()
        self.downsampling = downsampling   
        self.num_bins = num_bins
        self.input_dim = input_dim
        self.num_semantic_classes = num_semantic_classes

        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(70)
 
        self.scale_nets = nn.ModuleList()
        for i in range(num_semantic_classes):
            self.scale_nets.append(nn.Linear((70*70) + 4, self.num_bins + 1))

    def forward(self, x, boxes, labels):
        bins_map_scale = get_uniform_bins(torch.zeros(x.shape[0], 1, 1, 1).to(x.device), 0, 15, self.num_bins).squeeze(-1).squeeze(-1)

        boxes_valid_normalized_projected, num_valid_boxes = get_valid_normalized_projected_boxes(x, boxes, labels, self.downsampling)

        out = self.pool(F.relu(self.conv1(x)))

        out = torch.flatten(out, 1)
        out = torch.cat((out, boxes_valid_normalized_projected), dim=1) # Concat boxes

        out_list = []
        for i in range(self.num_semantic_classes):

            probs = torch.softmax(self.scale_nets[i](out), axis=1)
            scale = (probs * bins_map_scale.detach()).sum(1, keepdim=True)
       
            uncertainty = torch.sqrt((probs * ((bins_map_scale.detach() - scale)**2)).sum(1, keepdim=True))
            scale_unc = torch.cat([scale, uncertainty], dim=1)
            out_list.append(scale_unc)     

        out = torch.cat(out_list, dim=1)
        
        return out


class ROISelectScaleSmall(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectScaleSmall, self).__init__()
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)  
        self.pool = nn.AdaptiveAvgPool2d(45) 
        self.fc1 = nn.Linear((45*45) + 4, 2*num_semantic_classes) # Scale and shift 
        
    def forward(self, x, boxes, labels):
        boxes_valid_normalized_projected, num_valid_boxes = get_valid_normalized_projected_boxes(x, boxes, labels, self.downsampling)

        out = self.pool(F.relu(self.conv1(x)))

        out = torch.flatten(out, 1)
        out = torch.cat((out, boxes_valid_normalized_projected), dim=1) # Concat boxes

        out = self.fc1(out)
        out = out.view(num_valid_boxes, 2*self.num_semantic_classes)
        
        return out


"""
Predict scale with bins for a ROI with bbox concat
"""
class ROISelectScaleSmallBins(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=13, num_bins=50):
        super(ROISelectSmallBins, self).__init__()
        self.downsampling = downsampling   
        self.num_bins = num_bins
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(45)
 
        self.scale_nets = nn.ModuleList()
        for i in range(num_semantic_classes):
            self.scale_nets.append(nn.Linear((45*45) + 4, self.num_bins + 1))

    def forward(self, x, boxes, labels):
        bins_map_scale = get_uniform_bins(torch.zeros(x.shape[0], 1, 1, 1).to(x.device), 0, 15, self.num_bins).squeeze(-1).squeeze(-1)

        boxes_valid_normalized_projected, num_valid_boxes = get_valid_normalized_projected_boxes(x, boxes, labels, self.downsampling)

        out = self.pool(F.relu(self.conv1(x)))

        out = torch.flatten(out, 1)
        out = torch.cat((out, boxes_valid_normalized_projected), dim=1) # Concat boxes

        out_list = []
        for i in range(self.num_semantic_classes):

            probs = torch.softmax(self.scale_nets[i](out), axis=1)
            scale = (probs * bins_map_scale.detach()).sum(1, keepdim=True)
       
            uncertainty = torch.sqrt((probs * ((bins_map_scale.detach() - scale)**2)).sum(1, keepdim=True))
            scale_unc = torch.cat([scale, uncertainty], dim=1)
            out_list.append(scale_unc)     

        out = torch.cat(out_list, dim=1)
        
        return out


class ROISelectScaleBig(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_out=14):
        super(ROISelectScaleBig, self).__init__()
        self.downsampling = downsampling   
        self.num_out = num_out
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(input_dim, 96, 3, padding=1)  
        self.conv2 = nn.Conv2d(96, 1, 3, padding=1)  
        self.pool = nn.AdaptiveAvgPool2d(70) 
        self.fc1 = nn.Linear((70*70) + 4, num_out) # Scale and shift 

    def forward(self, x, boxes, labels):
        boxes_valid_normalized_projected, num_valid_boxes = get_valid_normalized_projected_boxes(x, boxes, labels, self.downsampling)

        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))

        out = torch.flatten(out, 1)
        out = torch.cat((out, boxes_valid_normalized_projected), dim=1)
        out = self.fc1(out)
        out = out.view(num_valid_boxes, self.num_out)
        
        return out


"""
Predict scale with bins for a ROI with bbox concat
"""
class ROISelectScaleBigBins(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=13, num_bins=50):
        super(ROISelectBigBins, self).__init__()
        self.downsampling = downsampling   
        self.num_bins = num_bins
        self.input_dim = input_dim
        self.num_semantic_classes = num_semantic_classes

        self.conv1 = nn.Conv2d(input_dim, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 1, 3, padding=1)  
        self.pool = nn.AdaptiveAvgPool2d(70)
 
        self.scale_nets = nn.ModuleList()
        for i in range(num_semantic_classes):
            self.scale_nets.append(nn.Linear((70*70) + 4, self.num_bins + 1))

    def forward(self, x, boxes, labels):
        bins_map_scale = get_uniform_bins(torch.zeros(x.shape[0], 1, 1, 1).to(x.device), 0, 15, self.num_bins).squeeze(-1).squeeze(-1)

        boxes_valid_normalized_projected, num_valid_boxes = get_valid_normalized_projected_boxes(x, boxes, labels, self.downsampling)

        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))

        out = torch.flatten(out, 1)
        out = torch.cat((out, boxes_valid_normalized_projected), dim=1) # Concat boxes

        out_list = []
        for i in range(self.num_semantic_classes):

            probs = torch.softmax(self.scale_nets[i](out), axis=1)
            scale = (probs * bins_map_scale.detach()).sum(1, keepdim=True)
       
            uncertainty = torch.sqrt((probs * ((bins_map_scale.detach() - scale)**2)).sum(1, keepdim=True))
            scale_unc = torch.cat([scale, uncertainty], dim=1)
            out_list.append(scale_unc)     

        out = torch.cat(out_list, dim=1)
        
        return out



class ROISelectScaleHuge(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_out=14):
        super(ROISelectScaleHuge, self).__init__()
        self.downsampling = downsampling   
        self.num_out = num_out
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(input_dim, input_dim, 3, padding=1)  
        self.conv2 = nn.Conv2d(input_dim, 96, 3, padding=1)  
        self.conv3 = nn.Conv2d(96, 1, 3, padding=1)  
        self.pool = nn.AdaptiveAvgPool2d(70) 
        self.fc1 = nn.Linear((70*70) + 4, 800)  
        self.fc2 = nn.Linear(800+4, num_out) # Scale and shift 
        
    def forward(self, x, boxes, labels):
        boxes_valid_normalized_projected, num_valid_boxes = get_valid_normalized_projected_boxes(x, boxes, labels, self.downsampling)

        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        out = self.pool(F.relu(self.conv3(out)))

        out = torch.flatten(out, 1)
        out = torch.cat((out, boxes_valid_normalized_projected), dim=1)
        out = self.fc1(out)
        out = torch.cat((out, boxes_valid_normalized_projected), dim=1)
        out = self.fc2(out)
        out = out.view(num_valid_boxes, num_out)
        
        return out


"""
Predict scale with bins for a ROI with bbox concat
"""
class ROISelectScaleHugeBins(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classesi=13, num_bins=50):
        super(ROISelectBigBins, self).__init__()
        self.downsampling = downsampling   
        self.num_bins = num_bins
        self.input_dim = input_dim
        self.num_semantic_classes = num_semantic_classes

        self.conv1 = nn.Conv2d(input_dim, input_dim, 3, padding=1)  
        self.conv2 = nn.Conv2d(input_dim, 96, 3, padding=1)  
        self.conv3 = nn.Conv2d(96, 1, 3, padding=1)   
        self.pool = nn.AdaptiveAvgPool2d(70)
 
        self.scale_nets_1 = nn.ModuleList()
        self.scale_nets_2 = nn.ModuleList()
        for i in range(num_semantic_classes):
            self.scale_nets_1.append(nn.Linear((70*70) + 4, 800))
            self.scale_nets_2.append(nn.Linear(800 + 4, self.num_bins + 1))

    def forward(self, x, boxes, labels):
        bins_map_scale = get_uniform_bins(torch.zeros(x.shape[0], 1, 1, 1).to(x.device), 0, 15, self.num_bins).squeeze(-1).squeeze(-1)

        boxes_valid_normalized_projected, num_valid_boxes = get_valid_normalized_projected_boxes(x, boxes, labels, self.downsampling)

        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        out = self.pool(F.relu(self.conv3(out)))

        out = torch.flatten(out, 1)
        out = torch.cat((out, boxes_valid_normalized_projected), dim=1) # Concat boxes

        out_list = []
        for i in range(self.num_semantic_classes):
            out = self.scale_nets_1[i](out)
            out = torch.cat((out, boxes_valid_normalized_projected), dim=1)
            out = self.scale_nets_2[i](out)

            probs = torch.softmax(out, axis=1)
            scale = (probs * bins_map_scale.detach()).sum(1, keepdim=True)
       
            uncertainty = torch.sqrt((probs * ((bins_map_scale.detach() - scale)**2)).sum(1, keepdim=True))
            scale_unc = torch.cat([scale, uncertainty], dim=1)
            out_list.append(scale_unc)     

        out = torch.cat(out_list, dim=1)
        
        return out


class ROISelectScaleA(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectScaleA, self).__init__()
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1) 
        self.fc1 = nn.Linear((120*160) + 4, 2*num_semantic_classes) # Scale and shift 

    def forward(self, x, boxes, labels):
        boxes_valid_normalized_projected, num_valid_boxes = get_valid_normalized_projected_boxes(x, boxes, labels, self.downsampling)

        out = F.relu(self.conv1(x))

        out = torch.flatten(out, 1)
        out = torch.cat((out, boxes_valid_normalized_projected), dim=1)
        out = self.fc1(out)
        out = out.view(num_valid_boxes, 2*self.num_semantic_classes)
        
        return out


class ROISelectScaleB(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectScaleB, self).__init__()
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(input_dim, 96, 3, padding=1)  
        self.conv2 = nn.Conv2d(96, 1, 3, padding=1)  
        self.fc1 = nn.Linear((120*160) + 4, 2*num_semantic_classes) # Scale and shift 
      
    def forward(self, x, boxes, labels):
        boxes_valid_normalized_projected, num_valid_boxes = get_valid_normalized_projected_boxes(x, boxes, labels, self.downsampling)

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))

        out = torch.flatten(out, 1)
        out = torch.cat((out, boxes_valid_normalized_projected), dim=1)
        out = self.fc1(out)
        out = out.view(num_valid_boxes, 2*self.num_semantic_classes)
        
        return out


class ROISelectScaleC(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectScaleC, self).__init__()
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(input_dim, 96, 3, padding=1)  
        self.conv2 = nn.Conv2d(96, 1, 3, padding=1)  
        self.fc1 = nn.Linear((120*160) + 4, 500)  
        self.fc2 = nn.Linear(500, 2*num_semantic_classes) # Scale and shift 
       
    def forward(self, x, boxes, labels):
        boxes_valid_normalized_projected, num_valid_boxes = get_valid_normalized_projected_boxes(x, boxes, labels, self.downsampling)

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))

        out = torch.flatten(out, 1)
        out = torch.cat((out, boxes_valid_normalized_projected), dim=1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        out = out.view(i, 2*self.num_semantic_classes)
        
        return out


"""
ROISelectScale end
"""


"""
ROISelectSharedCanonicalUniform start
"""


class ROISelectSharedCanonicalUniform(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=128, bin_num=40):
        super(ROISelectSharedCanonicalUniform, self).__init__()
          
        self.input_dim = input_dim
        self.bin_num = bin_num    

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, bin_num, 3, padding=1)

    def forward(self, x, boxes, labels):
        h, w = x.shape[2:]

        instances_per_batch, num_valid_boxes = get_valid_num_instances_per_batch(boxes, labels)

        out = F.relu(self.conv1(x))
        out = torch.softmax(self.conv2(out), 1)

        out = torch.cat([out[i, :, :, :].unsqueeze(0).repeat(times,1,1,1) for i, times in enumerate(instances_per_batch)], dim=0)        
        out = out.view(num_valid_boxes, self.bin_num, h, w)
        
        return out


class ROISelectSharedCanonicalBigUniform(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=128, bin_num=40):
        super(ROISelectSharedCanonicalBigUniform, self).__init__()
        
        self.input_dim = input_dim
        self.bin_num = bin_num

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, bin_num, 3, padding=1)

    def forward(self, x, boxes, labels):
        h, w = x.shape[2:]

        instances_per_batch, num_valid_boxes = get_valid_num_instances_per_batch(boxes, labels)

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = torch.softmax(self.conv3(out), 1)

        out = torch.cat([out[i, :, :, :].unsqueeze(0).repeat(times,1,1,1) for i, times in enumerate(instances_per_batch)], dim=0)
        out = out.view(num_valid_boxes, self.bin_num, h, w)
        
        return out


class ROISelectSharedCanonicalHugeUniform(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=128, bin_num=40):
        super(ROISelectSharedCanonicalHugeUniform, self).__init__()
              
        self.input_dim = input_dim
        self.bin_num = bin_num

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv5 = nn.Conv2d(hidden_dim, bin_num, 3, padding=1)

    def forward(self, x, boxes, labels):
        h, w = x.shape[2:]

        instances_per_batch, num_valid_boxes = get_valid_num_instances_per_batch(boxes, labels)

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(x))
        out = F.relu(self.conv3(x))
        out = F.relu(self.conv4(x))
        out = torch.softmax(self.conv5(out), 1)

        out = torch.cat([out[i, :, :, :].unsqueeze(0).repeat(times,1,1,1) for i, times in enumerate(instances_per_batch)], dim=0)
        out = out.view(num_valid_boxes, self.bin_num, h, w)
        
        return out


"""
ROISelectSharedCanonicalUniform end
"""


"""
ROISelectCanonical start
"""


class ROISelectCanonical(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectCanonical, self).__init__()
              
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(input_dim+4, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, num_semantic_classes, 3, padding=1)

    def forward(self, x, boxes, labels):

        h, w = x.shape[2:]

        boxes_valid_normalized_projected, num_valid_boxes = get_valid_normalized_projected_boxes(x, boxes, labels, self.downsampling)

        boxes_valid_normalized_projected = boxes_valid_normalized_projected.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)
        
        out = torch.cat((x, boxes_valid_normalized_projected), dim=1)

        out = F.relu(self.conv1(out))
        out = torch.sigmoid(self.conv2(out))

        out = out.view(num_valid_boxes, self.num_semantic_classes, h, w)
        
        return out


class ROISelectCanonicalA(nn.Module):
    def __init__(self, input_dim=32, num_semantic_classes=14):
        super(ROISelectCanonicalA, self).__init__()
              
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(input_dim, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, num_semantic_classes, 3, padding=1)

    def forward(self, x, boxes, labels):

        h, w = x.shape[2:]

        _, num_valid_boxes = get_valid_boxes(boxes, labels)


        out = F.relu(self.conv1(x))
        out = torch.sigmoid(self.conv2(out))

        out = out.view(num_valid_boxes, self.num_semantic_classes, h, w)
        
        return out


class ROISelectCanonicalB(nn.Module):
    def __init__(self, input_dim=32, num_semantic_classes=14):
        super(ROISelectCanonicalB, self).__init__()
              
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(input_dim, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, num_semantic_classes, 3, padding=1)


    def forward(self, x, boxes, labels):

        h, w = x.shape[2:]
        
        _, num_valid_boxes = get_valid_boxes(boxes, labels)

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = torch.sigmoid(self.conv3(out))

        out = out.view(num_valid_boxes, self.num_semantic_classes, h, w)
        
        return out


class ROISelectCanonicalC(nn.Module):
    def __init__(self, input_dim=32, num_semantic_classes=14):
        super(ROISelectCanonicalC, self).__init__()
              
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(input_dim, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 96, 3, padding=1)
        self.conv4 = nn.Conv2d(96, num_semantic_classes, 3, padding=1)

    def forward(self, x, boxes, labels):

        h, w = x.shape[2:]

        _, num_valid_boxes = get_valid_boxes(boxes, labels)

        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = torch.sigmoid(self.conv4(out))

        out = out.view(num_valid_boxes, self.num_semantic_classes, h, w)
        
        return out


class ROISelectCanonicalD(nn.Module):
    def __init__(self, input_dim=32, num_semantic_classes=14):
        super(ROISelectCanonicalD, self).__init__()
              
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(input_dim, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, num_semantic_classes, 3, padding=1)

    def forward(self, x, boxes, labels):

        h, w = x.shape[2:]
        _, num_valid_boxes = get_valid_boxes(boxes, labels)

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = torch.sigmoid(self.conv3(out))

        out = out.view(num_valid_boxes, self.num_semantic_classes, h, w)
        
        return out


"""
ROISelectCanonical end
"""


"""
ROISelectSharedCanonical start
"""


class ROISelectSharedCanonical(nn.Module):
    def __init__(self, input_dim=32, num_semantic_classes=14):
        super(ROISelectSharedCanonical, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(input_dim, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, num_semantic_classes, 3, padding=1)

    def forward(self, x, boxes, labels):

        h, w = x.shape[2:]
        
        instances_per_batch, num_valid_boxes = get_valid_num_instances_per_batch(boxes, labels)
        
        out = F.relu(self.conv1(x))
        out = torch.sigmoid(self.conv2(out))
        
        out = torch.cat([out[i, :, :, :].unsqueeze(0).repeat(times,1,1,1) for i, times in enumerate(instances_per_batch)], dim=0)
        out = out.view(num_valid_boxes, 1, h, w)
        
        return out


class ROISelectSharedCanonicalBig(nn.Module):
    def __init__(self, input_dim=32, num_semantic_classes=14):
        super(ROISelectSharedCanonicalBig, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(input_dim, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, num_semantic_classes, 3, padding=1)

    def forward(self, x, boxes, labels):

        h, w = x.shape[2:]
        
        instances_per_batch, num_valid_boxes = get_valid_num_instances_per_batch(boxes, labels)

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = torch.sigmoid(self.conv3(out))
        
        out = torch.cat([out[i, :, :, :].unsqueeze(0).repeat(times,1,1,1) for i, times in enumerate(instances_per_batch)], dim=0)
        out = out.view(num_valid_boxes, 1, h, w)
        
        return out


class ROISelectSharedCanonicalHuge(nn.Module):
    def __init__(self, input_dim=32, num_semantic_classes=14):
        super(ROISelectSharedCanonicalHuge, self).__init__()
              
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(input_dim, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, num_semantic_classes, 3, padding=1)

    def forward(self, x, boxes, labels):

        h, w = x.shape[2:]

        instances_per_batch, num_valid_boxes = get_valid_num_instances_per_batch(boxes, labels)

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = torch.sigmoid(self.conv5(out))

        out = torch.cat([out[i, :, :, :].unsqueeze(0).repeat(times,1,1,1) for i, times in enumerate(instances_per_batch)], dim=0)
        out = out.view(num_valid_boxes, 1, h, w)
        
        return out


"""
ROISelectSharedCanonical end
"""


"""
ROISelectSharedCanonicalClass start
"""


class ROISelectCanonicalSharedClass(nn.Module):
    def __init__(self, input_dim=32, num_semantic_classes=14):
        super(ROISelectCanonicalSharedClass, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(input_dim, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, num_semantic_classes, 3, padding=1)

    def forward(self, x):
        
        out = F.relu(self.conv1(x))
        out = torch.sigmoid(self.conv2(out))

        return out

class ROISelectCanonicalSharedClassBig(nn.Module):
    def __init__(self, input_dim=32, num_semantic_classes=14):
        super(ROISelectCanonicalSharedClassBig, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(input_dim, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, num_semantic_classes, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = torch.sigmoid(self.conv3(out))
        return out


"""
ROISelectSharedCanonicalClass end
"""


"""
ROISelectScaleModule start
"""


class ROISelectScaleModule(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectScaleModule, self).__init__()
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1) 
        self.pool = nn.AdaptiveAvgPool2d(70) 
        self.fc1 = nn.Linear((70*70) + 4, 2*num_semantic_classes) # Scale and shift 

    def forward(self, x, boxes, labels, class_label):

        with torch.no_grad(): 
            h, w = x.shape[2:]

            batch_size, num_max_instances = boxes.shape[0:2]

            boxes_reshaped = boxes.view(batch_size * num_max_instances, 4)

            labels_reshaped = labels.view(batch_size * num_max_instances, 1)
            labels_valid = torch.nonzero(labels_reshaped == class_label)
            boxes_valid = boxes_reshaped[labels_valid[:, 0]]

            num_valid_boxes = boxes_valid.shape[0]

            boxes_valid_projected = project_box_to_feature_map(boxes_valid, self.downsampling)
            boxes_valid_projected_normalized = normalize_box(boxes_valid_projected, height=h, width=w)

        out = self.pool(F.relu(self.conv1(x)))
        out = torch.flatten(out, 1)
        
        out = torch.cat((out, boxes_valid_projected_normalized), dim=1)
        out = self.fc1(out)

        out = out.view(num_valid_boxes, 2*self.num_semantic_classes)

        return out


class ROISelectScaleModuleBig(nn.Module):
    def __init__(self, input_dim=32, downsampling=4, num_semantic_classes=14):
        super(ROISelectScaleModuleBig, self).__init__()

        
        self.downsampling = downsampling   
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(input_dim, input_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(input_dim, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, 3, padding=1) 
        self.pool = nn.AdaptiveAvgPool2d(70) 
        self.fc1 = nn.Linear((70*70) + 4, 2*num_semantic_classes) 

    def forward(self, x, boxes, labels, class_label):

        with torch.no_grad(): 
            h, w = x.shape[2:]

            batch_size, num_max_instances = boxes.shape[:2]

            boxes_reshaped = boxes.view(batch_size * num_max_instances, 4)

            labels_reshaped = labels.view(batch_size * num_max_instances, 1)
            labels_valid = torch.nonzero(labels_reshaped == class_label)
            boxes_valid = boxes_reshaped[labels_valid[:, 0]]

            num_valid_boxes = boxes_valid.shape[0]

            boxes_valid_projected = project_box_to_feature_map(boxes_valid, self.downsampling)
            boxes_valid_projected_normalized = normalize_box(boxes_valid_projected, height=h, width=w)

        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        out = self.pool(F.relu(self.conv3(out)))

        out = torch.flatten(out, 1)
        out = torch.cat((out, boxes_valid_projected_normalized), dim=1)
        out = self.fc1(out)

        out = out.view(num_valid_boxes, 2*self.num_semantic_classes)

        return out


"""
ROISelectScaleModule end
"""


"""
ROISelectCanonicalModule start
"""
class ROISelectCanonicalModule(nn.Module):
    def __init__(self, input_dim=32, num_semantic_classes=14):
        super(ROISelectCanonicalModule, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(input_dim, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, num_semantic_classes, 3, padding=1)

    def forward(self, x, boxes, labels, class_label):
        with torch.no_grad(): 
            h, w = x.shape[2:]
            batch_size, num_max_instances = boxes.shape[:2]

            boxes_reshaped = boxes.view(batch_size * num_max_instances, 4)

            # Pick instances of given class #
            valid_boxes = labels.view(batch_size * num_max_instances, 1)
            valid_boxes = torch.nonzero(valid_boxes == class_label)
            boxes_valid = boxes_reshaped[valid_boxes[:, 0]]

            num_valid_boxes = boxes_valid.shape[0]

        out = F.relu(self.conv1(x))
        out = torch.sigmoid(self.conv2(out))
        out = out.view(num_valid_boxes, self.num_semantic_classes, h, w)
        
        return out

class ROISelectCanonicalModuleBig(nn.Module):
    def __init__(self, input_dim=32, num_semantic_classes=14):
        super(ROISelectCanonicalModuleBig, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(input_dim, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, num_classes, 3, padding=1)

    def forward(self, x, boxes, labels, class_label):
        with torch.no_grad(): 
            h, w = x.shape[2:]
            batch_size, num_max_instances = boxes.shape[:2]

            boxes_reshaped = boxes.view(batch_size * num_max_instances, 4)

            # Pick instances of given class #
            valid_boxes = labels.view(batch_size * num_max_instances, 1)
            valid_boxes = torch.nonzero(valid_boxes == class_label)
            boxes_valid = boxes_reshaped[valid_boxes[:, 0]]

            num_valid_boxes = boxes_valid.shape[0]

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = torch.sigmoid(self.conv2(out))

        out = out.view(num_valid_boxes, self.num_semantic_classes, h, w)
        
        return out


"""
ROISelectCanonicalModule end
"""


"""
Other
"""


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
