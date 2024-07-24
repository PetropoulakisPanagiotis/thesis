import json
import numpy as np
import cv2
import os
import time
import g2o

from collections import defaultdict, namedtuple

from threading import Thread, Lock
from multiprocessing import Process, Queue


class ScaleReader(object):
    def __init__(self, ids, timestamps=None, min_var=1e-4, max_var=50):
        self.ids = ids
        self.timestamps = timestamps
        self.max_var = max_var
        self.min_var = min_var
        self.cache = dict()

        self.preload()

    def preload(self):
        for idx, id in enumerate(self.ids):
            scale, scale_unc, scale_type = [], [], []
            with open(id, 'r') as json_file:
                data = json.load(json_file)
                scale = data['scale']  # This is a list
                scale_unc = np.clip(np.asarray(data['scale_uncertainty'], dtype=np.float32), a_min=self.min_var,
                                    a_max=self.max_var).tolist()
                scale_type = data['scale_type']
                scale_valid = data['valid'] if not scale_type == 'single' else [1]
                self.cache[idx] = (scale, scale_unc, scale_type, scale_valid)

    def read(self, path):
        scale, scale_unc, scale_type = [], [], []
        with open(path, 'r') as json_file:
            data = json.load(json_file)
            scale = data['scale']  # This is a list
            scale_unc = np.clip(np.asarray(data['scale_uncertainty'], dtype=np.float32), a_min=self.min_var,
                                a_max=self.max_var).tolist()
            scale_type = data['scale_type']
            scale_valid = data['valid'] if not scale_type == 'single' else [1]

        return scale, scale_unc, scale_type, scale_valid

    def __getitem__(self, idx):
        scale, scale_unc, scale_type, scale_valid = self.cache[idx]
        #scale, scale_unc, scale_type, scale_valid = self.read(self.ids[idx])
        return scale, scale_unc, scale_type, scale_valid

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        for i, timestamp in enumerate(self.timestamps):
            yield timestamp, self[i]

    @property
    def dtype(self):
        return self[0].dtype

    @property
    def shape(self):
        return self[0].shape

class GtPoseReader(object):
    def __init__(self, ids, timestamps=None):
        self.ids = ids
        self.timestamps = timestamps
        self.cache = dict()

        self.preload()

    def preload(self):
        for idx, id in enumerate(self.ids):
            with open(id, 'r') as json_file:
                data = json.load(json_file)
                x, y, z = data['x'], data['y'], data['z']
                qx, qy, qz, qw = data['quat_x'], data['quat_y'], data['quat_z'], data['quat_w']
            
                gt_pose = g2o.Isometry3d(g2o.Quaternion([qx, qy, qz, qw]), [x, y, z])
      
                self.cache[idx] = gt_pose

    def read(self, path):
        with open(path, 'r') as json_file:
            data = json.load(json_file)
            x, y, z = data['x'], data['y'], data['z']
            qx, qy, qz = data['quat_x'], data['quat_y'], data['quat_z']
        
            gt_pose = g2o.Isometry3d(g2o.Quaternion(qx, qy, qz), [x, y, z])
  
        return gt_pose

    def __getitem__(self, idx):
        gt_pose = self.cache[idx]
        #gt_pose = self.read(self.ids[idx])
        return gt_pose

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        for i, timestamp in enumerate(self.timestamps):
            yield timestamp, self[i]

    @property
    def dtype(self):
        return g2o.Isometry3d

    @property
    def shape(self):
        return (1, 6)


class UncertaintyReader(object):
    def __init__(self, ids, timestamps=None, min_var=1e-4, max_var=50, optimization_type='per_class'):
        self.ids = ids
        self.timestamps = timestamps
        self.max_var = max_var
        self.min_var = min_var
        self.optimization_type = optimization_type
        self.cache = dict()

        self.preload()

    def preload(self):
        for idx, id in enumerate(self.ids):
            canonical_uncertainty = np.load(id)
            if 'per_class' or 'per_instance' in self.optimization_type:
                canonical_uncertainty = canonical_uncertainty.squeeze(0).squeeze(0)

            self.cache[idx] = np.clip(canonical_uncertainty, self.min_var, self.max_var)

    def read(self, path):
        canonical_uncertainty = np.load(path)
        canonical_uncertainty = np.clip(canonical_uncertainty, self.min_var, self.max_var)
        if 'per_class' or 'per_instance' in self.optimization_type:
            canonical_uncertainty = canonical_uncertainty.squeeze(0).squeeze(0)

        return canonical_uncertainty

    def __getitem__(self, idx):
        canonical_uncertainty = self.cache[idx]
        #canonical_uncertainty = self.read(self.ids[idx])
        return canonical_uncertainty

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        for i, timestamp in enumerate(self.timestamps):
            yield timestamp, self[i]

    @property
    def dtype(self):
        return self[0].dtype

    @property
    def shape(self):
        return self[0].shape


class ImageReader(object):
    def __init__(self, ids, timestamps=None, cam=None):
        self.ids = ids
        self.timestamps = timestamps
        self.cam = cam
        self.cache = dict()
        self.idx = 0

        self.ahead = 10  # 10 images ahead of current index
        self.waiting = 1.5  # waiting time

        self.preload_thread = Thread(target=self.preload)
        self.thread_started = False

    def read(self, path):
        img = cv2.imread(path, -1)
        if self.cam is None:
            return img
        else:
            return self.cam.rectify(img)

    def preload(self):
        idx = self.idx
        t = float('inf')
        while True:
            if time.time() - t > self.waiting:
                return
            if self.idx == idx:
                time.sleep(1e-2)
                continue

            for i in range(self.idx, self.idx + self.ahead):
                if i not in self.cache and i < len(self.ids):
                    self.cache[i] = self.read(self.ids[i])
            if self.idx + self.ahead > len(self.ids):
                return
            idx = self.idx
            t = time.time()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        self.idx = idx
        #if not self.thread_started:
        #    self.thread_started = True
        #    self.preload_thread.start()

        if idx in self.cache:
            img = self.cache[idx]
            del self.cache[idx]
        else:
            img = self.read(self.ids[idx])
        return img

    def __iter__(self):
        for i, timestamp in enumerate(self.timestamps):
            yield timestamp, self[i]

    @property
    def dtype(self):
        return self[0].dtype

    @property
    def shape(self):
        return self[0].shape


class ICLNUIMDataset(object):
    '''
    path example: 'path/to/your/ICL-NUIM R-GBD Dataset/living_room_traj0_frei_png'
    '''

    cam = namedtuple('camera', 'fx fy cx cy scale')(481.20, 480.0, 319.5, 239.5, 5000)

    def __init__(self, path):
        path = os.path.expanduser(path)
        self.rgb = ImageReader(self.listdir(os.path.join(path, 'rgb')))
        self.depth = ImageReader(self.listdir(os.path.join(path, 'depth')))
        self.timestamps = None

    def sort(self, xs):
        return sorted(xs, key=lambda x: int(x[:-4]))

    def listdir(self, dir):
        files = [_ for _ in os.listdir(dir) if _.endswith('.png')]
        return [os.path.join(dir, _) for _ in self.sort(files)]

    def __len__(self):
        return len(self.rgb)


def make_pair(matrix, threshold=1):
    assert (matrix >= 0).all()
    pairs = []
    base = defaultdict(int)
    while True:
        i = matrix[:, 0].argmin()
        min0 = matrix[i, 0]
        j = matrix[0, :].argmin()
        min1 = matrix[0, j]

        if min0 < min1:
            i, j = i, 0
        else:
            i, j = 0, j
        if min(min1, min0) < threshold:
            pairs.append((i + base['i'], j + base['j']))

        matrix = matrix[i + 1:, j + 1:]
        base['i'] += (i + 1)
        base['j'] += (j + 1)

        if min(matrix.shape) == 0:
            break
    return pairs


def associate(first_list, second_list, offset=0, max_difference=0.02):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim 
    to find the closest match for every input tuple.

    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))

    """
    first_keys = first_list.keys()
    second_keys = second_list.keys()
    potential_matches = [(abs(a - (b + offset)), a, b) for a in first_keys for b in second_keys
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            #first_keys.remove(a)
            #second_keys.remove(b)
            matches.append((a, b))

    matches.sort()
    return matches


class TUMRGBDDataset(object):
    '''
    path example: 'path/to/your/TUM R-GBD Dataset/rgbd_dataset_freiburg1_xyz'
    '''

    cam = namedtuple('camera', 'fx fy cx cy scale')(525.0, 525.0, 319.5, 239.5, 5000)

    def __init__(self, path):
        path = os.path.expanduser(path)
        with open(path + "/associations.txt", 'r') as file:
            entries = file.readlines()

        timestamps_rgb = []
        timestamps_depth = []
        rgb_ids = []
        depth_ids = []
        for entry in entries:
            entry = entry.split(" ")
            timestamps_rgb.append(float(entry[0]))
            timestamps_depth.append(float(entry[2]))
            rgb_ids.append(path + entry[1])
            depth_ids.append(path + entry[3].rstrip('\n'))

        self.rgb = ImageReader(rgb_ids, timestamps_rgb)
        self.depth = ImageReader(depth_ids, timestamps_depth)
        self.timestamps = timestamps_rgb

    def __getitem__(self, idx):
        return self.rgb[idx], self.depth[idx]

    def __len__(self):
        return len(self.rgb)


class ScanNetDataset(object):
    '''
    path example: 'path/to/your/TUM R-GBD Dataset/rgbd_dataset_freiburg1_xyz'
    '''
    cam = None

    # valid #
    cam = namedtuple('camera', 'fx fy cx cy scale')(577.5906635802469, 576.3481987847223, 319.15804639274694,
                                                    241.9392752941744, 1000)

    def __init__(self, path, scene='scene0191_00', split='train', scale_aware=True, optimization_type='global',
                 max_var=50.0, min_var=1e-4, network_depth=True, total=None, local=False, debug=True):
        self.scale_aware = scale_aware
        path = os.path.expanduser(path)
        self.max_var = max_var
        self.min_var = min_var

        with open(path + "/" + split + '/rgb_intrinsics/' + scene + ".json", 'r') as json_file:
            data = json.load(json_file)
            ScanNetDataset.cam = namedtuple('camera', 'fx fy cx cy scale scale_unc')(data['fx'], data['fy'], data['cx'],
                                                                                     data['cy'], 1000.0, 10000.0)

        ids = [(file.split('.')[0]) for file in os.listdir(path + "/" + split + "/rgb/" + scene)]
        ids = sorted(ids, key=lambda x: str(x))

        rgb_timestamps = [float(id) for id in ids]
        self.timestamps = rgb_timestamps
        depth_timestamps = rgb_timestamps
        rgb_ids = [path + '/' + split + '/rgb/' + scene + "/" + str(item) + '.jpg' for item in ids]

        if self.scale_aware:
            if local:
                depth_ids = [
                    path + '/' + split + '/predictions/' + scene + '/' + optimization_type + "/depth/" + str(item) +
                    '.png' for item in ids
                ]
                canonical_ids = [
                    path + '/' + split + '/predictions/' + scene + '/' + optimization_type + "/canonical/" + str(item) +
                    '.png' for item in ids
                ]
                canonical_unc_ids = [
                    path + '/' + split + '/predictions/' + scene + '/' + optimization_type + "/canonical_unc/" +
                    str(item) + '.npy' for item in ids
                ]
                scales_ids = [
                    path + '/' + split + '/predictions/' + scene + '/' + optimization_type + "/scale/" + str(item) +
                    '.json' for item in ids
                ]
                pixel_to_scale_map_ids = [
                    path + '/' + split + '/predictions/' + scene + '/' + optimization_type + "/scale_map/" + str(item) +
                    '.png' for item in ids
                ]
            else:
                path = path.replace('/scannet/data_converted', '')
                depth_ids = [
                    path + '/predictions/' + scene + '/' + optimization_type + "/depth/" + str(item) + '.png'
                    for item in ids
                ]
                canonical_ids = [
                    path + '/predictions/' + scene + '/' + optimization_type + "/canonical/" + str(item) + '.png'
                    for item in ids
                ]
                canonical_unc_ids = [
                    path + '/predictions/' + scene + '/' + optimization_type + "/canonical_unc/" + str(item) + '.npy'
                    for item in ids
                ]
                scales_ids = [
                    path + '/predictions/' + scene + '/' + optimization_type + "/scale/" + str(item) + '.json'
                    for item in ids
                ]
                pixel_to_scale_map_ids = [
                    path + '/predictions/' + scene + '/' + optimization_type + "/scale_map/" + str(item) + '.png'
                    for item in ids
                ]

        else:
            if network_depth:
                if local:
                    depth_ids = [
                        path + '/' + split + '/predictions/' + scene + '/' + optimization_type + "/depth/" + str(item) +
                        '.png' for item in ids
                    ]
                else:
                    path = path.replace('/scannet/data_converted', '')
                    depth_ids = [
                        path + '/predictions/' + scene + '/' + optimization_type + "/depth/" + str(item) + '.png'
                        for item in ids
                    ]
            else:
                depth_ids = [path + '/' + split + '/depth/' + scene + "/" + str(item) + '.png' for item in ids]

        if debug:
            gt_ids = [path + '/scannet/data_converted' + '/' + split + '/extrinsics/' + scene + "/" + str(item) + '.json' for item in ids]
            self.gt_poses = GtPoseReader(gt_ids, rgb_timestamps) 
            
            seg_masks_ids = [path + '/scannet/data_converted' + '/' + split + '/semantic_refined_20/' + scene + "/" + str(item) + '.png' for item in ids]
            self.sem_masks = ImageReader(seg_masks_ids, rgb_timestamps) 

        if total is not None:
            rgb_ids = rgb_ids[:total]
            depth_ids = depth_ids[:total]
            if self.scale_aware:
                canonical_ids = canonical_ids[:total]
                canonical_unc_ids = canonical_unc_ids[:total]
                pixel_to_scale_map_ids = pixel_to_scale_map_ids[:total]
                scales_ids = scales_ids[:total]

        # Read depth #
        self.rgb = ImageReader(rgb_ids, rgb_timestamps)
        self.depth = ImageReader(depth_ids, depth_timestamps)
        if self.scale_aware:
            # Canonical #
            self.canonical = ImageReader(canonical_ids, depth_timestamps)
            self.canonical_uncertainty = UncertaintyReader(canonical_unc_ids, rgb_timestamps, self.min_var,
                                                           self.max_var, optimization_type)

            # Map #
            self.pixel_to_scale_map = ImageReader(pixel_to_scale_map_ids, depth_timestamps)

            # Scale #
            self.scale = ScaleReader(scales_ids, rgb_timestamps, self.min_var, self.max_var)

    def __getitem__(self, idx):
        return self.rgb[idx], self.depth[idx]

    def __len__(self):
        return len(self.rgb)
