import cv2
import numpy as np
import json
import os
from collections import namedtuple
from scipy.spatial.transform import Rotation as R



def visualize_depth_map(depth_map: np.ndarray) -> None:
    # Normalize depth values to the range [0, 255]
    min_depth = np.min(depth_map[depth_map > 0])  # Ignore zero (invalid depth)
    max_depth = np.max(depth_map)
    norm_depth = (depth_map - min_depth) / (max_depth - min_depth)
    norm_depth = (norm_depth * 255).astype(np.uint8)

    # Apply a colormap: closer points are darker, distant points are lighter
    colormap = cv2.COLORMAP_JET
    depth_colored = cv2.applyColorMap(norm_depth, colormap)

    # Display the depth map
    cv2.imshow("Depth Map Visualization", depth_colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_test_points_pixel_and_world_coords(cam: namedtuple, camera_to_world_transform: np.ndarray, depth_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pass


if __name__ == '__main__':
    path = '/home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted'
    scene = 'scene0655_01'
    split = 'valid'
    test_id = 5

    ids = [(file.split('.')[0]) for file in os.listdir(path + "/" + split + "/rgb/" + scene)]
    ids = sorted(ids, key=lambda x: str(x))

    rgb_timestamps = [float(id) for id in ids]
    depth_timestamps = rgb_timestamps
    rgb_ids = [path + '/' + split + '/rgb/' + scene + "/" + str(item) + '.jpg' for item in ids]
    depth_ids = [path + '/' + split + '/depth/' + scene + "/" + str(item) + '.png' for item in ids]

    img_test_id_str = ids[test_id]
    img_test_path = rgb_ids[test_id]
    depth_test_path = depth_ids[test_id]

    img_base_id_str = ids[0]
    img_base_path = ids[0]

    # Intrinsics #
    with open(path + "/" + split + '/rgb_intrinsics/' + scene + ".json", 'r') as json_file:
        data = json.load(json_file)
        cam = namedtuple('camera', 'fx fy cx cy scale')(
        data['fx'],
        data['fy'],
        data['cx'],
        data['cy'],
        1000.0)

    # Extrinsics test image #
    with open(path + "/" + split + '/extrinsics/' + scene + "/" + img_test_id_str + ".json", 'r') as json_file:
        data = json.load(json_file)
        qx_test, qy_test, qz_test, qw_test = data['quat_x'], data['quat_y'], data['quat_z'], data['quat_w']
        x_test, y_test, z_test = data['x'], data['y'], data['z']

    rotation_test = R.from_quat([qx_test, qy_test, qz_test, qw_test])
    transformation_matrix_test = np.eye(4)
    transformation_matrix_test[:3, :3] = rotation_test.as_matrix()
    transformation_matrix_test[:3, 3] = [x_test, y_test, z_test]

    # Extrinsics base image #
    with open(path + "/" + split + '/extrinsics/' + scene + "/" + img_base_path + ".json", 'r') as json_file:
        data = json.load(json_file)
        qx_base, qy_base, qz_base, qw_base = data['quat_x'], data['quat_y'], data['quat_z'], data['quat_w']
        x_base, y_base, z_base = data['x'], data['y'], data['z']

    rotation_base = R.from_quat([qx_base, qy_base, qz_base, qw_base])
    transformation_matrix_base = np.eye(4)
    transformation_matrix_base[:3, :3] = rotation_base.as_matrix()
    transformation_matrix_base[:3, 3] = [x_base, y_base, z_base]
    transformation_matrix_base_inv = transformation_matrix_base.T

    # Base frame is not at zero, transform frames to the origin #
    transformation_matrix_test_corrected = transformation_matrix_base_inv * transformation_matrix_test

    # Load test image and depth #
    img_test = cv2.imread(img_test_path, -1)
    depth_test = cv2.imread(depth_test_path, -1)
    depth_test = np.asarray(depth_test)
    depth_test = depth_test / cam.scale

    visualize_depth_map(depth_test)
