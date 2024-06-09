import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os
import json

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

# Define the dataset class
class MyDataset(Dataset):
    def __init__(self, original_data_path, refined_data_path):
        self.original_data_path = original_data_path
        self.refined_data_path = refined_data_path

        self.optimized_scale_folder = os.path.join(self.refined_data_path, 'optimized_scale/')
        self.scale_map_folder = os.path.join(self.original_data_path, 'scale_map/')
        self.canonical_folder = os.path.join(original_data_path, 'canonical/')
        self.depth_folder = os.path.join(original_data_path, 'depth/')

        self.optimized_scale = sorted([f for f in os.listdir(self.optimized_scale_folder) if f.endswith('.json')])

        self.canonical = [f for f in os.listdir(self.canonical_folder) if f.endswith('.png') and f.split('.')[0] + '.json' in self.optimized_scale]
        self.depth = [f for f in os.listdir(self.depth_folder) if f.endswith('.png') and f.split('.')[0] + '.json' in self.optimized_scale]
        self.scale_map = [f for f in os.listdir(self.scale_map_folder) if f.endswith('.png') and f.split('.')[0] + '.json' in self.optimized_scale]

        for scale in self.optimized_scale:
            if scale.split('.')[0] + '.png' not in self.canonical:
                print(scale)

        assert len(self.optimized_scale) == len(self.depth) and len(self.optimized_scale) == len(self.depth) \
               and len(self.optimized_scale) == len(self.scale_map)

    def __getitem__(self, index):

        scale_file = os.path.join(self.optimized_scale_folder, self.optimized_scale[index])
        # Read optimized scale #
        with open(scale_file) as file:
            data = json.load(file)
            scale = data['scale'] # This is a list

        scale = torch.Tensor(np.asarray(scale)).type(torch.float32)

        # Read scale map #
        scale_map_file = os.path.join(self.scale_map_folder, self.scale_map[index])
        scale_map = cv2.imread(scale_map_file, -1)
        scale_map = torch.Tensor(scale_map).type(torch.int64)

        # Read canonical and depth #
        canonical_file = os.path.join(self.canonical_folder, self.canonical[index])
        canonical = cv2.imread(canonical_file, -1) / 1000 
        canonical = torch.Tensor(canonical).type(torch.float32)

        depth_file = os.path.join(self.depth_folder, self.depth[index])
        depth = cv2.imread(depth_file, -1) / 1000 
        depth = torch.Tensor(depth).type(torch.float32)

        # Calculated refined depth #
        depth_refined = canonical * torch.take(scale, scale_map.flatten()).view(canonical.shape)

        data = {"depth": depth, "depth_refined": depth_refined}
        return data

    def __len__(self):
        return len(self.optimized_scale)

if __name__ == '__main__':

    min_depth = 0.1
    max_depth = 10

    # Define the data loader
    dataset = MyDataset(original_data_path='/home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted/valid/network_predictions/scene0655_01/segmentation', \
                        refined_data_path='/home/petropoulakis/Desktop/thesis/code/thesis/rgbd_ptam/results/segmentation')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

    # Iterate over the data in a loop
    for i, batch in enumerate(data_loader):
        # Do something with the batch (e.g., pass it to a model)
        print(batch)
        exit()
