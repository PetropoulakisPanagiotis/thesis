import os
import json
from scipy.spatial.transform import Rotation as R
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

                    """
                    rotation_matrix = R.from_quat([quat_x, quat_y, quat_z, quat_w]).as_matrix()
                    transformation_matrix = np.eye(4)
                    transformation_matrix[:3, :3] = rotation_matrix
                    transformation_matrix[:3, 3] = [x, y, z]
                    transformation_matrix = np.linalg.inv(transformation_matrix)

                    r = R.from_matrix(transformation_matrix[:3, :3])
                    quat = r.as_quat()
                    quat_x = quat[0]
                    quat_y = quat[1]
                    quat_z = quat[2]
                    quat_w = quat[3]
                    translation = transformation_matrix[:3, 3]
                    x = translation[0]
                    y = translation[1]
                    z = translation[2]
                    """

                    entry = (id_, x, y, z, quat_x, quat_y, quat_z, quat_w)  # Using tuple for each entry
                    entries.append(entry)

    # Sort the entries by ID
    entries.sort(key=lambda x: x[0])
    return entries

def convet_scannet_to_tum(path, output_file):
    entries = read_scannet_json_files(path)
    with open(output_file, 'w') as txt_file:
        for entry in entries:
            txt_file.write(' '.join(map(str, entry)) + '\n')  # Joining the tuple elements with space and writing to file

# Directory containing JSON files
#directory = '/home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted/train/extrinsics/scene0191_00/'
#directory = '/home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted/valid/extrinsics/scene0568_00/'
#directory = '/home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted/valid/extrinsics/scene0568_01/'
#directory = '/home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted/valid/extrinsics/scene0412_00/'
#directory = '/home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted/valid/extrinsics/scene0164_00/'
#directory = '/home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted/valid/extrinsics/scene0608_00/'
#directory = '/home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted/valid/extrinsics/scene0084_02/'

# 2
directory = '/home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted/valid/extrinsics/scene0025_02/'
# 1
directory = '/home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted/valid/extrinsics/scene0655_01/'

# Output text file
#output_file = '/home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted/slam_gt_train.txt'
#output_file = '/home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted/slam_gt_valid.txt'
output_file = '/home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted/slam_gt_valid.txt'

# Write data to text file
convet_scannet_to_tum(directory, output_file)
