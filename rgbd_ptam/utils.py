import os
import json
import numpy as np


def read_scannet_json_files(directory):
    entries = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                    id_ = int(file.split('.')[0])
                    x = data.get('x')
                    y = data.get('y')
                    z = data.get('z')
                    quat_x = data.get('quat_x')
                    quat_y = data.get('quat_y')
                    quat_z = data.get('quat_z')
                    quat_w = data.get('quat_w')

                    entry = (id_, x, y, z, quat_x, quat_y, quat_z, quat_w)  # Using tuple for each entry
                    entries.append(entry)

    # Sort the entries by ID
    entries.sort(key=lambda x: x[0])
    return entries


def convet_scannet_to_tum(path, output_file):
    entries = read_scannet_json_files(path)
    with open(output_file, 'w') as txt_file:
        for entry in entries:
            txt_file.write(' '.join(map(str, entry)) +
                           '\n')  # Joining the tuple elements with space and writing to file


#directory_gt = '/home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted/valid/extrinsics/'
#save_dir = '/home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted/valid/gt_traj/'

directory_gt = '/usr/stud/petp/storage/user/petp/datasets/scannet/data_converted/valid/extrinsics/'
save_dir = '/usr/stud/petp/storage/user/petp/datasets/scannet/data_converted/valid/gt_traj/'

#scenes = ['scene0655_01']
scenes = ['scene0025_01', 'scene0568_02', 'scene0153_00', 'scene0527_00', 'scene0086_02', 'scene0684_00', 'scene0314_00', 'scene0558_02', 'scene0100_02', 
    'scene0685_01', 'scene0693_01', 'scene0664_02', 'scene0553_00', 'scene0064_00', 'scene0647_00',
    'scene0609_03',
    'scene0574_01',
    'scene0300_01',
    'scene0598_02',
    'scene0019_00', 
    'scene0063_00',
    'scene0077_00',
    'scene0081_00', 
    'scene0131_02',
    'scene0193_01',	
    'scene0164_01', 
    'scene0221_01', 
    'scene0277_01', 
    'scene0278_01', 
    'scene0316_00', 
    'scene0338_02', 
    'scene0342_00', 
    'scene0356_02', 
    'scene0377_02', 
    'scene0382_01',
                ]


for scene in scenes:
    gt_path = directory_gt + scene

    out_file = save_dir + scene
    os.makedirs(out_file, exist_ok=True)
    out_file += '/gt_traj.txt'

    # Write data to text file
    convet_scannet_to_tum(gt_path, out_file)
