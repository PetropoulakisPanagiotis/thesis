import cv2
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os
import json
import pandas as pd

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

def find_set_of_common_files(path, ext_str):
    local_path = path
    global_set = set()
    per_class_set = set()
    per_instance_set = set()
    if 'global' in path:
        global_set = set([f for f in os.listdir(path) if f.endswith(ext_str)])

        local_path = local_path.replace('global', 'per_class')
        per_class_set = set(os.listdir(local_path))

        local_path = local_path.replace('per_class', 'per_instance')
        per_instance_set = set(os.listdir(local_path))
    elif 'per_class' in path:
        per_class_set = set([f for f in os.listdir(path) if f.endswith(ext_str)])

        local_path = local_path.replace('per_class', 'global')
        global_set = set(os.listdir(local_path))

        local_path = local_path.replace('global', 'per_instance')
        per_instance_set = set(os.listdir(local_path))
    else:
        per_instance_set = set([f for f in os.listdir(path) if f.endswith(ext_str)])

        local_path = local_path.replace('per_instance', 'global')
        global_set = set(os.listdir(local_path))

        local_path = local_path.replace('global', 'per_class')
        per_class_set = set(os.listdir(local_path))

    common_set = global_set & per_class_set & per_instance_set
    return common_set

# Define the dataset class
class MyDataset(Dataset):
    def __init__(self, true_depth_path, original_data_path, refined_data_path):
        self.mask_to_mappoints = False
        self.use_original_depth = False
        self.original_data_path = original_data_path
        self.refined_data_path = refined_data_path

        self.optimized_scale_folder = os.path.join(self.refined_data_path, 'optimized_scale/')
        self.optimized_scale_map_folder = os.path.join(self.refined_data_path, 'optimized_scale/')

        self.scale_map_folder = os.path.join(self.original_data_path, 'scale_map/')
        self.original_scale_folder = os.path.join(self.original_data_path, 'scale/')
        self.canonical_folder = os.path.join(original_data_path, 'canonical/')
        self.depth_folder = os.path.join(original_data_path, 'depth/')

        self.true_depth_folder = true_depth_path

        self.optimized_scale = sorted(find_set_of_common_files(self.optimized_scale_folder, '.json'))
        self.optimized_scale_map = sorted(find_set_of_common_files(self.optimized_scale_folder, 'scale_map.png'))

        self.size_dataset = len(self.optimized_scale)
        assert self.size_dataset == len(self.optimized_scale_map)

        self.canonical = sorted([f for f in os.listdir(self.canonical_folder) if f.endswith('.png') and f.split('.')[0] + '.json' in self.optimized_scale])
        self.depth = sorted([f for f in os.listdir(self.depth_folder) if f.endswith('.png') and f.split('.')[0] + '.json' in self.optimized_scale])
        self.scale_map = sorted([f for f in os.listdir(self.scale_map_folder) if f.endswith('.png') and f.split('.')[0] + '.json' in self.optimized_scale])
        self.original_scale = sorted([f for f in os.listdir(self.original_scale_folder) if f.endswith('.json') and f.split('.')[0] + '.json' in self.optimized_scale])

        self.true_depth = sorted([f for f in os.listdir(self.true_depth_folder) if f.endswith('.png') and f.split('.')[0] + '.json' in self.optimized_scale])

        assert self.size_dataset == len(self.depth) and self.size_dataset == len(self.depth) \
               and self.size_dataset == len(self.scale_map) and self.size_dataset == len(self.true_depth) \
               and self.size_dataset == len(self.original_scale)

    def __getitem__(self, index):

        scale_file = os.path.join(self.optimized_scale_folder, self.optimized_scale[index])
        # Read optimized scale #
        with open(scale_file) as file:
            data = json.load(file)
            scale = data['scale'] # This is a list

        scale = torch.Tensor(np.asarray(scale)).type(torch.float32)

        original_scale_file = os.path.join(self.original_scale_folder, self.original_scale[index])
        # Read optimized scale #
        with open(original_scale_file) as file:
            data = json.load(file)
            original_scale = data['scale'] # This is a list

        original_scale = torch.Tensor(np.asarray(original_scale)).type(torch.float32)

        # Read scale map #
        scale_map_file = os.path.join(self.scale_map_folder, self.scale_map[index])
        scale_map = cv2.imread(scale_map_file, -1)
        scale_map = torch.Tensor(scale_map).type(torch.int64)

        optimized_scale_map_file = os.path.join(self.optimized_scale_map_folder, self.optimized_scale_map[index])
        optimized_scale_map = cv2.imread(optimized_scale_map_file, - 1)
        optimized_scale_map = torch.Tensor(optimized_scale_map).type(torch.int64)

        # Read canonical and depth #
        canonical_file = os.path.join(self.canonical_folder, self.canonical[index])
        canonical = cv2.imread(canonical_file, -1) / 5000 
        canonical = torch.Tensor(canonical).type(torch.float32)

        depth_file = os.path.join(self.depth_folder, self.depth[index])
        depth = cv2.imread(depth_file, -1) / 5000 
        depth = torch.Tensor(depth).type(torch.float32)

        true_depth_file = os.path.join(self.true_depth_folder, self.true_depth[index])
        true_depth = cv2.imread(true_depth_file, -1) / 1000 
        true_depth = torch.Tensor(true_depth).type(torch.float32)

        # Calculated refined depth - whole depth map using scale #
        depth_refined = canonical * torch.take(scale, scale_map.flatten()).view(canonical.shape)
        if self.use_original_depth == False:
            depth = canonical * torch.take(original_scale, scale_map.flatten()).view(canonical.shape)

        # Select only depth of optimized mappoints #
        if self.mask_to_mappoints:
            canonical = canonical * optimized_scale_map
            depth = depth * optimized_scale_map
            true_depth = true_depth * optimized_scale_map

        data = {"true_depth": true_depth, "depth": depth, "depth_refined": depth_refined}
        return data

    def __len__(self):
        return self.size_dataset

if __name__ == '__main__':

    min_depth = 0.1
    max_depth = 10


    true_depth_dir = '/usr/stud/petp/storage/user/petp/datasets/scannet/data_converted/valid/depth/'
    refined_depth_dir = '/usr/stud/petp/code/thesis/rgbd_ptam/results_slam/'
    original_network_depth_dir = '/usr/stud/petp/storage/user/petp/datasets/predictions_final_slam/'
    scenes = [
        'scene0664_02',
        'scene0314_00',
        'scene0064_00',
        'scene0086_02',
        'scene0598_02',
        'scene0574_01',
        'scene0685_01',
        'scene0300_01',
        'scene0527_00',
        'scene0684_00',
        'scene0019_00',
        'scene0193_01',
        'scene0131_02',
        'scene0025_01',
        'scene0221_01',
        'scene0164_01',
        'scene0316_00',
        'scene0693_01',
        'scene0100_02',
        'scene0609_03',
        'scene0553_00',
        'scene0342_00',
        'scene0081_00',
        'scene0278_01',
    ]   
        
    scenes = ['scene0338_02',
        'scene0019_00',
        'scene0316_00',
        'scene0693_01',
        'scene0609_03',
        'scene0278_01',
]



    variations = ['per_instance', 'per_class', 'global']
    runs = 3

    save_dir = './results_best_refined_new/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    columns = ['scene', 'per_instace', 'per_class', 'global']

    df_results = pd.DataFrame(columns=columns)
    for scene in tqdm(scenes, total=len(scenes), desc='scenes'):
        true_depth_path = true_depth_dir + scene

        main_metric_per_variation = []
        for variation in tqdm(variations, total=len(variations), desc='variations'):
            original_data_path = original_network_depth_dir + scene + '/' + variation

            total_errors_network = []
            total_errors_refined = [] 
            for run in tqdm(range(runs), total=runs, desc='runs'):
                refined_data_path = refined_depth_dir + scene + '/' + variation + '/' + variation + '_' + str(run)

                # Define the data loader
                dataset = MyDataset(true_depth_path=true_depth_path, original_data_path=original_data_path, refined_data_path=refined_data_path)
                data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

                errors_network = np.zeros(9)
                errors_refined = np.zeros(9)
                # Iterate over the data in a loop
                for i, batch in tqdm(enumerate(data_loader), total=len(dataset), desc='frames'):
                    # Do something with the batch (e.g., pass it to a model)
                    true_depth = batch["true_depth"].cpu().numpy()
                    depth = batch["depth"].cpu().numpy()
                    depth_refined = batch["depth_refined"].cpu().numpy()

                    true_depth = true_depth.flatten()
                    true_depth[np.isinf(true_depth)] = min_depth
                    true_depth[np.isnan(true_depth)] = max_depth

                    depth = depth.flatten()
                    depth[np.isinf(depth)] = min_depth
                    depth[np.isnan(depth)] = max_depth
                    depth[depth < min_depth] = min_depth
                    depth[depth > max_depth] = max_depth

                    depth_refined = depth_refined.flatten()
                    depth_refined[depth_refined < min_depth] = min_depth
                    depth_refined[depth_refined > max_depth] = max_depth
                    depth_refined[np.isinf(depth_refined)] = min_depth
                    depth_refined[np.isnan(depth_refined)] = max_depth

                    valid_depth = np.logical_and(true_depth >= min_depth, true_depth <= max_depth)

                    errors_network = errors_network + np.asarray(compute_errors(true_depth[valid_depth], depth[valid_depth]))
                    errors_refined = errors_refined + np.asarray(compute_errors(true_depth[valid_depth], depth_refined[valid_depth]))
                
                errors_network /= len(dataset)
                errors_refined /= len(dataset)

                total_errors_network.append(errors_network)
                total_errors_refined.append(errors_refined)

            total_errors_network = np.sum(np.asarray(total_errors_network), axis=0) / runs
            total_errors_refined = np.sum(np.asarray(total_errors_refined), axis=0) / runs

            if  total_errors_network[3] >= total_errors_refined[3]:
                percentage_better = (total_errors_network[3] - total_errors_refined[3]) / total_errors_refined[3] * 100
            else:               
                percentage_better = (total_errors_refined[3] - total_errors_network[3]) / total_errors_network[3] * 100
                percentage_better *= -1

            main_metric_per_variation.append(percentage_better) # negative means the refined are worst
 
            if True:
                print('Computing errors for {} eval samples'.format(len(dataset)))
                print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                             'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                             'd3'))
                print(total_errors_network)
                print(total_errors_refined)

        main_metric_per_variation.insert(0, scene)
        df_results = pd.concat([df_results, pd.DataFrame([main_metric_per_variation], columns=columns)], ignore_index=True)

    df_results.to_csv(save_dir + 'refined_results.csv', index=False)
