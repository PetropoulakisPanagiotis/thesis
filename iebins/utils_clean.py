import os, sys
import math

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Sampler
from torchvision import transforms

SCANNET_COLOR_MAP_20 = {
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    14: (247., 182., 210.),
    15: (66., 188., 102.),
    16: (219., 219., 141.),
    17: (140., 57., 197.),
    18: (202., 185., 52.),
    19: (51., 176., 203.),
    20: (200., 54., 131.),
    21: (92., 193., 61.),
    22: (78., 71., 183.),
    23: (172., 114., 82.),
    24: (255., 127., 14.),
    25: (91., 163., 138.),
    26: (153., 98., 156.),
    27: (140., 153., 101.),
    28: (158., 218., 229.),
    29: (100., 125., 154.),
    30: (178., 127., 135.),
    32: (146., 111., 194.),
    33: (44., 160., 44.),
    34: (112., 128., 144.),
    35: (96., 207., 209.),
    36: (227., 119., 194.),
    37: (213., 92., 176.),
    38: (94., 106., 211.),
    39: (82., 84., 163.),
    40: (100., 85., 144.),
}

# set missing colors in dictionary, for source of colors see:
# https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/util.py
SCANNET_COLOR_MAP_20[13] = (178, 76, 76)
SCANNET_COLOR_MAP_20[31] = (120, 185, 128)


CLASS_LABELS_20 = ('void', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                   'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                   'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture')


VALID_CLASS_IDS_20 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)


SEMANTIC_CLASS_COLORS_SCANNET_20 = tuple(
            tuple(int(val) for val in SCANNET_COLOR_MAP_20[idx])  # convert float to int
            for idx in ((0,) + (VALID_CLASS_IDS_20))
        )


color_mappings = {}
for i in range(len(SEMANTIC_CLASS_COLORS_SCANNET_20)):
    color_mappings[i] = SEMANTIC_CLASS_COLORS_SCANNET_20[i]


inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']


cmap_name = "custom_colormap"
colors = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]  
custom_cmap_labels = LinearSegmentedColormap.from_list(cmap_name, colors, N=14)
custom_cmap_instances = LinearSegmentedColormap.from_list(cmap_name, colors, N=40)


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def block_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


def get_num_lines(file_path):
    f = open(file_path, 'r')

    lines = f.readlines()
    f.close()

    return len(lines)


def load_checkpoint_custom(checkpoint_path, gpu, retrain, model, optimizer):
    if checkpoint_path != '':
        if os.path.isfile(checkpoint_path):
            print("== Loading checkpoint '{}'".format(checkpoint_path))
            if gpu is None:
                checkpoint = torch.load(checkpoint_path)
            else:
                loc = 'cuda:{}'.format(gpu)
                checkpoint = torch.load(checkpoint_path, map_location=loc)
           
            # Skip weights of some layers #
            weights_to_remove_1 = "update"
            weights_to_remove_2 = "project"
            keys_to_remove = [key for key in checkpoint['model'].keys() if weights_to_remove_1 in key]
            keys_to_remove.extend([key for key in checkpoint['model'].keys() if weights_to_remove_2 in key]) 

            for key_to_remove in keys_to_remove:
                checkpoint['model'].pop(key_to_remove)

            model.load_state_dict(checkpoint['model'], strict=False)
            #optimizer.load_state_dict(checkpoint['optimizer'])

            if not retrain:
                try:
                    global_step = checkpoint['global_step']
                    best_eval_measures_higher_better = checkpoint['best_eval_measures_higher_better'].cpu()
                    best_eval_measures_lower_better = checkpoint['best_eval_measures_lower_better'].cpu()
                    best_eval_steps = checkpoint['best_eval_steps']
                except KeyError:
                    print("Could not load values for online evaluation")

            print("== Loaded checkpoint '{}' (global_step {})".format(checkpoint_path, checkpoint['global_step']))
        
            del checkpoint
        else:
            print("== No checkpoint found at '{}'".format(checkpoint_path))

def load_checkpoint_custom_full_model(checkpoint_path, gpu, retrain, model, optimizer):
    if checkpoint_path != '':
        if os.path.isfile(checkpoint_path):
            print("== Loading checkpoint '{}'".format(checkpoint_path))
            if gpu is None:
                checkpoint = torch.load(checkpoint_path)
            else:
                loc = 'cuda:{}'.format(gpu)
                checkpoint = torch.load(checkpoint_path, map_location=loc)

            model.load_state_dict(checkpoint['model'], strict=False)
            print("== Loaded checkpoint '{}' (global_step {})".format(checkpoint_path, checkpoint['global_step']))
        
            del checkpoint
        else:
            print("== No checkpoint found at '{}'".format(checkpoint_path))

def set_hparams_dict(args, num_semantic_classes):
    hparams = {
        "epochs": args.num_epochs,
        "loss_type": args.loss_type,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "update_block": args.update_block,
        "max_tree_depth": args.max_tree_depth,
        "bin_num": args.bin_num,
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
        "train_decoder": args.train_decoder,
        "predict_unc": args.predict_unc,
        "num_semantic_classes": num_semantic_classes,
        "segmentation": args.segmentation,
        "instances": args.instances,
        "var": args.var,
        "padding_instances": args.padding_instances,
    }

    return hparams
        

def colorize(value, vmin=None, vmax=None, cmap='Greys'):
    value = value.cpu().numpy()[:, :, :]
    value = np.log10(value)

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.0

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
        value = value * 0.0

    return np.expand_dims(value, 0)


def compute_errors(gt, pred, var=None):
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

    if var is not None:
        cons_unc = (gt - pred) ** 2 / var 
        cons_unc = np.mean(cons_unc)
        return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3, cons_unc]

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]


def compute_error_uncertainty(gt, pred, unc):
    u_gt = np.exp(-5 * np.abs(gt - pred) / (gt + pred + 1e-7))
    unc_error = np.abs(unc - u_gt).mean()
    
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


class d3vo_loss(nn.Module):
    def __init__(self, beta=0.5):
        super(d3vo_loss, self).__init__()
        self.beta = beta

    def forward(self, depth_est, depth_gt, unc, mask):
        return torch.mean((((torch.abs(depth_est[mask] - depth_gt[mask]) / unc[mask]) + torch.log(unc[mask])) + (math.log(2 * math.pi))) * (unc[mask].detach() ** self.beta))


def sigma_metric_d3vo(depth_c, unc_c, scale, unc_scale, args):
    
    if args.segmentation:

        # It is std, make it variance 
        unc_c = unc_c ** 2

        sigma_metric = depth_c ** 2 * unc_scale.unsqueeze(-1).unsqueeze(-1) + scale.unsqueeze(-1).unsqueeze(-1) ** 2 * depth_c
    return sigma_metric

def entropy_loss(preds, gt_label, mask):
    # preds: B, C, H, W
    # gt_label: B, H, W
    # mask: B, H, W
    mask = mask > 0.0             
    preds = preds.permute(0, 2, 3, 1) # B, H, W, C
    preds_mask = preds[mask]          # N, C
    gt_label_mask = gt_label[mask]    # N
    loss = F.cross_entropy(preds_mask, gt_label_mask, reduction='mean')
    
    return loss


def colormap(inputs, name='jet', normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    _DEPTH_COLORMAP = plt.get_cmap(name, 256)   
    
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
        image : torch.Tensor [B,3,H,W]
    Returns
        image_flipped : torch.Tensor [B,3,H,W]
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
    """
    Sampler that restricts data loading to a subset of the dataset.

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
    
    
def find_indexes_valid_instances(labels):
    """
    1-dim
    """
    return torch.nonzero(labels != 0).squeeze()
