import os, sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Sampler
from torchvision import transforms

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)


eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']

colors = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]  # Replace with your desired colors
n_bins = [15]  # Number of bins, can be adjusted based on your data

cmap_name = "custom_colormap"
custom_cmap_labels = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins[0])
custom_cmap_instances = LinearSegmentedColormap.from_list(cmap_name, colors, N=40)

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


def block_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def colorize(value, vmin=None, vmax=None, cmap='Greys'):
    value = value.cpu().numpy()[:, :, :]
    value = np.log10(value)

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value*0.

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)

    img = value[:, :, :3]

    return img.transpose((2, 0, 1))


def normalize_result(value, vmin=None, vmax=None):
    value = value.cpu().numpy()[0, :, :]

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.

    return np.expand_dims(value, 0)


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]

def compute_errors_uncertainty(gt, pred, unc, type_unc, beta=0.5):
    if type_unc == 0:
        u_gt = np.exp(-5 * np.abs(gt - pred) / (gt + pred + 1e-7))
        unc_error = np.abs(unc - u_gt) 
    else: 
        unc_error = (((np.abs(pred - gt) / unc) + np.log(unc)) + (math.log(2 * math.pi))) * (unc ** beta)
    
    return unc_error

class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0

class l1_loss(nn.Module):
    def __init__(self):
        super(l1_loss, self).__init__()

    def forward(self, depth_est, depth_gt, mask):
        return torch.mean(torch.abs(depth_est[mask] - depth_gt[mask]))

def entropy_loss(preds, gt_label, mask):
    # preds: B, C, H, W
    # gt_label: B, H, W
    # mask: B, H, W
    mask = mask > 0.0 # B, H, W
    preds = preds.permute(0, 2, 3, 1) # B, H, W, C
    preds_mask = preds[mask] # N, C
    gt_label_mask = gt_label[mask] # N
    loss = F.cross_entropy(preds_mask, gt_label_mask, reduction='mean')
    return loss


def colormap(inputs, name='jet', normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()
    _DEPTH_COLORMAP = plt.get_cmap(name, 256)  # for plotting
    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis[0,:,:,:]


def flip_lr(image):
    """
    Flip image horizontally

    Parameters
    ----------
    image : torch.Tensor [B,3,H,W]
        Image to be flipped

    Returns
    -------
    image_flipped : torch.Tensor [B,3,H,W]
        Flipped image
    """
    assert image.dim() == 4, 'You need to provide a [B,C,H,W] image to flip'
    
    return torch.flip(image, [3])


def fuse_inv_depth(inv_depth, inv_depth_hat, method='mean'):
    """
    Fuse inverse depth and flipped inverse depth maps

    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map
    inv_depth_hat : torch.Tensor [B,1,H,W]
        Flipped inverse depth map produced from a flipped image
    method : str
        Method that will be used to fuse the inverse depth maps

    Returns
    -------
    fused_inv_depth : torch.Tensor [B,1,H,W]
        Fused inverse depth map
    """
    if method == 'mean':
        return 0.5 * (inv_depth + inv_depth_hat)
    elif method == 'max':
        return torch.max(inv_depth, inv_depth_hat)
    elif method == 'min':
        return torch.min(inv_depth, inv_depth_hat)
    else:
        raise ValueError('Unknown post-process method {}'.format(method))


def post_process_depth(depth, depth_flipped, method='mean'):
    """
    Post-process an inverse and flipped inverse depth map

    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map
    inv_depth_flipped : torch.Tensor [B,1,H,W]
        Inverse depth map produced from a flipped image
    method : str
        Method that will be used to fuse the inverse depth maps

    Returns
    -------
    inv_depth_pp : torch.Tensor [B,1,H,W]
        Post-processed inverse depth map
    """
    B, C, H, W = depth.shape
    inv_depth_hat = flip_lr(depth_flipped)
    inv_depth_fused = fuse_inv_depth(depth, inv_depth_hat, method=method)
    
    xs = torch.linspace(0., 1., W, device=depth.device, dtype=depth.dtype).repeat(B, C, H, 1)
    mask = 1.0 - torch.clamp(20. * (xs - 0.05), 0., 1.)
    mask_hat = flip_lr(mask)

    return mask_hat * depth + mask * inv_depth_hat + (1.0 - mask - mask_hat) * inv_depth_fused


class DistributedSamplerNoEvenlyDivisible(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        num_samples = int(math.floor(len(self.dataset) * 1.0 / self.num_replicas))
        rest = len(self.dataset) - num_samples * self.num_replicas
        if self.rank < rest:
            num_samples += 1
     
        self.num_samples = num_samples
        self.total_size = len(dataset)
        # self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]
        # assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)
        # assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
    
    
class D_to_cloud(nn.Module):
    """Layer to transform depth into point cloud
    """
    def __init__(self, batch_size, height, width):
        super(D_to_cloud, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32) # 2, H, W    
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords), requires_grad=False) # 2, H, W  

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False) # B, 1, H, W

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0) # 1, 2, L
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1) # B, 2, L
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1), requires_grad=False) # B, 3, L

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points

        return cam_points.permute(0, 2, 1)


def find_indexes_valid_instances(labels):
    return torch.nonzero(labels!=0).squeeze()

def debug_result(result, gt_depth):
    if True: 
        if True:
            print("depth")
            print(torch.max(result['pred_depths_r_list'][-1][0, :, :, :]))
            print(torch.min(result['pred_depths_r_list'][-1][0, :, :, :]))
        if True:
            print("canonical")
            print(torch.max(result['pred_depths_rc_list'][-1][:, :, :, :]))
            print(torch.min(result['pred_depths_rc_list'][-1][:, :, :, :]))
            print(torch.mean(result['pred_depths_rc_list'][-1][:, :, :, :]))
            print(torch.std(result['pred_depths_rc_list'][-1][:, :, :, :]))
        if False:
            print("uncertainty (std)")
            print(torch.max(result['uncertainty_maps_list'][-1][0, 0, :, :]))
            print(torch.min(result['uncertainty_maps_list'][-1][0, 0, :, :]))
            print(torch.mean(result['uncertainty_maps_list'][-1][0, 0, :, :]))
            print(torch.std(result['uncertainty_maps_list'][-1][0, 0, :, :]))            
        if True:
            print("scale")
            print(torch.max(result['pred_scale_list'][-1][:, :]))
            print(torch.min(result['pred_scale_list'][-1][:, :]))
            print(torch.mean(result['pred_scale_list'][-1][:, :]))
            print(torch.std(result['pred_scale_list'][-1][:, :]))            
        if True:
            print("shift")
            print(torch.max(result['pred_shift_list'][-1][:, :]))
            print(torch.min(result['pred_shift_list'][-1][:, :]))
            print(torch.mean(result['pred_shift_list'][-1][:, :]))
            print(torch.std(result['pred_shift_list'][-1][:, :]))
        if False:
            print("sample")
            print(result['pred_depths_r_list'][-1][0, 0, 120, 701])
            print(gt_depth[0, 120, 701, 0])
        if False:
            print("instances scale")
            print(torch.max(result['pred_scale_instances_list'][-1][0, 0]))
            print(torch.min(result['pred_scale_instances_list'][-1][0, 0]))
            print(torch.mean(result['pred_scale_instances_list'][-1][0, 0]))
            print(torch.std(result['pred_scale_instances_list'][-1][0, 0]))  
            print("instances shift")
            print(torch.max(result['pred_shift_instances_list'][-1][0, 0]))
            print(torch.min(result['pred_shift_instances_list'][-1][0, 0]))
            print(torch.mean(result['pred_shift_instances_list'][-1][0, 0]))
            print(torch.std(result['pred_shift_instances_list'][-1][0, 0]))  
            print("instances canonical")
            print(torch.max(result['pred_depths_instances_rc_list'][-1][0, 0, :, :]))
            print(torch.min(result['pred_depths_instances_rc_list'][-1][0, 0, :, :]))
            print(torch.mean(result['pred_depths_instances_rc_list'][-1][0, 0, :, :]))
            print(torch.std(result['pred_depths_instances_rc_list'][-1][0, 0, :, :]))

"""Train parser"""
train_parser = argparse.ArgumentParser(description='Scale PyTorch implementation.', fromfile_prefix_chars='@')
train_parser.convert_arg_line_to_args = convert_arg_line_to_args

train_parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
train_parser.add_argument('--model_name',                type=str,   help='model name', default='nyu')
train_parser.add_argument('--encoder',                   type=str,   help='type of encoder, base07, large07, tiny07', default='tiny07')
train_parser.add_argument('--pretrain',                  type=str,   help='path of pretrained encoder', default=None)

# Dataset
train_parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
train_parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
train_parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', required=True)
train_parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
train_parser.add_argument('--input_height',              type=int,   help='input height', default=480)
train_parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
train_parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)
train_parser.add_argument('--min_depth',                 type=float, help='minimum depth in estimation', default=0.1)

# Bins 
train_parser.add_argument('--update_block',              type=int,   help='update block: iebins (0), canonical one scale per pixel (1),  with uncertainty prediction (from GRU) (2), # Canonical - one scale with uncertainty (from decoder) concatenation (3), Canonical. one scale per image (4)', default='1')
train_parser.add_argument('--var',                       type=int,   help='Variation of instances block', default='0')
train_parser.add_argument('--padding_instances',         type=int,   help='How many pixels to padd for box instances', default='0')
train_parser.add_argument('--max_tree_depth',            type=int,   help='max GRU iterations', default='6')
train_parser.add_argument('--bin_num',                   type=int,   help='number of bins', default='16')
train_parser.add_argument('--predict_unc',               dest='predict_unc',help='True to predict uncertainty from the decoder', action='store_true')

train_parser.add_argument('--segmentation',              dest='segmentation', help='segmentation variation', action='store_true')
train_parser.add_argument('--instances',                 dest='instances', help='instances variation', action='store_true')

# Log and save
train_parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
train_parser.add_argument('--exp_name',                  type=str,   help='directory to save checkpoints and summaries', default='exp-1')
train_parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
train_parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=100)
train_parser.add_argument('--save_freq',                 type=int,   help='Checkpoint saving frequency in global steps', default=5000)

# Training
train_parser.add_argument('--train_decoder',             type=int,   help='how many layers to train from the decoder', default=1)
train_parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
train_parser.add_argument('--loss_type',                 type=int,   help='0 for silog, 1 for l1', default=0)
train_parser.add_argument('--uncertainty_weight',        type=float, help='weight for uncertainty loss', default=1), 
train_parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
train_parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
train_parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
train_parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
train_parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
train_parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)
train_parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)

# Preprocessing
train_parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
train_parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
train_parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
train_parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

# Multi-gpu training
train_parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)
train_parser.add_argument('--world_size',                type=int,   help='number of nodes for distributed training', default=1)
train_parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)
train_parser.add_argument('--dist_url',                  type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
train_parser.add_argument('--dist_backend',              type=str,   help='distributed backend', default='nccl')
train_parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default=None)
train_parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
                                                                    'N processes per node, which has N GPUs. This is the '
                                                                    'fastest way to use PyTorch for either single node or '
                                                                    'multi node data parallel training', action='store_true',)
# Online eval
train_parser.add_argument('--do_online_eval',                        help='if set, perform online eval in every eval_freq steps', action='store_true')
train_parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', required=False)
train_parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=False)
train_parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', required=False)
train_parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
train_parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
train_parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
train_parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
train_parser.add_argument('--eval_freq',                 type=int,   help='Online evaluation frequency in global steps', default=500)
train_parser.add_argument('--eval_summary_directory',    type=str,   help='output directory for eval summary,'
                                                                    'if empty outputs to checkpoint folder', default='')
"""Eval parser"""
eval_parser = argparse.ArgumentParser(description='Scale-Aware PyTorch implementation.', fromfile_prefix_chars='@')
eval_parser.convert_arg_line_to_args = convert_arg_line_to_args

eval_parser.add_argument('--model_name',                type=str,   help='model name', default='iebins')
eval_parser.add_argument('--encoder',                   type=str,   help='type of encoder, base07, large07, tiny07', default='large07')
eval_parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
eval_parser.add_argument('--pretrain',                  type=str,   help='path of pretrained encoder', default=None)

eval_parser.add_argument('--segmentation',              dest='segmentation', help='segmentation variation', action='store_true')
eval_parser.add_argument('--instances',                 dest='instances', help='instances variation', action='store_true')
eval_parser.add_argument('--var',                       type=int,   help='Variation of instances block', default='0')
eval_parser.add_argument('--padding_instances',         type=int,   help='How many pixels to padd for box instances', default='0')

# Dataset
eval_parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
eval_parser.add_argument('--input_height',              type=int,   help='input height', default=480)
eval_parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
eval_parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)
eval_parser.add_argument('--min_depth',                 type=float, help='minimum depth in estimation', default=0)

# Bins
eval_parser.add_argument('--update_block',              type=int,   help='update block: iebins (0), canonical one scale per pixel (1),  with uncertainty prediction (from GRU) (2), # Canonical - one scale with uncertainty (from decoder) concatenation (3), Canonical. one scale per image (4)', default='0')
eval_parser.add_argument('--max_tree_depth',            type=int,   help='max GRU iterations', default='6')
eval_parser.add_argument('--bin_num',                   type=int,   help='number of bins', default='16')
eval_parser.add_argument('--predict_unc',               dest='predict_unc',help='True to predict uncertainty from the decoder', action='store_true')

# Preprocessing
eval_parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
eval_parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
eval_parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
eval_parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

# Eval
eval_parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='.')
eval_parser.add_argument('--exp_name',                  type=str,   help='directory to save checkpoints and summaries', default='exp-1')
eval_parser.add_argument('--data_path_eval',            type=str,   help='path to the data for evaluation', required=False)
eval_parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
eval_parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for evaluation', required=False)
eval_parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for evaluation', required=False)
eval_parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
eval_parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
eval_parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
eval_parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')

eval_parser.add_argument('--loss_type',                 type=int,   help='0 for silog, 1 for l1', default=0)
eval_parser.add_argument('--pick_class',                type=int,   help='Evaluate single class for debug', default=3)
