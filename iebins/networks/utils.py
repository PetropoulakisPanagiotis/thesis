import torch
import torch.nn as nn
import torch.nn.functional as F

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

def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """
    Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
