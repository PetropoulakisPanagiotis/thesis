import json
import numpy as np
import cv2
import os
import time

from collections import defaultdict, namedtuple

from threading import Thread, Lock
from multiprocessing import Process, Queue



class ImageReader(object):
    def __init__(self, ids, timestamps=None, cam=None):
        self.ids = ids
        self.timestamps = timestamps
        self.cam = cam
        self.cache = dict()
        self.idx = 0

        self.ahead = 10      # 10 images ahead of current index
        self.waiting = 1.5   # waiting time

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
        # if not self.thread_started:
        #     self.thread_started = True
        #     self.preload_thread.start()

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

    cam = namedtuple('camera', 'fx fy cx cy scale')(
        481.20, 480.0, 319.5, 239.5, 5000)
    def __init__(self, path):
        path = os.path.expanduser(path)
        self.rgb = ImageReader(self.listdir(os.path.join(path, 'rgb')))
        self.depth = ImageReader(self.listdir(os.path.join(path, 'depth')))
        self.timestamps = None

    def sort(self, xs):
        return sorted(xs, key=lambda x:int(x[:-4]))

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

def associate(first_list, second_list,offset=0,max_difference=0.02):
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
    potential_matches = [(abs(a - (b + offset)), a, b) 
                         for a in first_keys 
                         for b in second_keys 
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

    cam = namedtuple('camera', 'fx fy cx cy scale')(
        525.0, 525.0, 319.5, 239.5, 5000)

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
    cam = namedtuple('camera', 'fx fy cx cy scale')(
        577.5906635802469, 576.3481987847223, 319.15804639274694, 241.9392752941744, 1000)

    def __init__(self, path, scene='scene0191_00', split='train'):
        path = os.path.expanduser(path)



        with open(path + "/" + split + '/rgb_intrinsics/' + scene + ".json", 'r') as json_file:
            data = json.load(json_file)
            ScanNetDataset.cam = namedtuple('camera', 'fx fy cx cy scale')(
            data['fx'],
            data['fy'],
            data['cx'],
            data['cy'],
            1000.0)

        ids = [(file.split('.')[0]) for file in os.listdir(path + "/" + split + "/rgb/" + scene)]
        ids = sorted(ids, key=lambda x: str(x))

        rgb_timestamps = [float(id) for id in ids]
        depth_timestamps = rgb_timestamps
        rgb_ids = [path + '/' + split + '/rgb/' + scene + "/" + str(item) + '.jpg' for item in ids]
        #depth_ids = [path + '/' + split + '/depth/' + scene + "/" + str(item) + '.png' for item in ids]
        depth_ids = [path + '/' + split + '/depth_network/' + scene + "/" + str(item) + '.png' for item in ids]

        # Read depth #
        self.rgb = ImageReader(rgb_ids, rgb_timestamps)
        self.depth = ImageReader(depth_ids, depth_timestamps)
        self.timestamps = rgb_timestamps

    def __getitem__(self, idx):
        return self.rgb[idx], self.depth[idx]

    def __len__(self):
        return len(self.rgb)
