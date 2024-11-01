import cv2
import numpy as np
import random
import json
import os
from collections import namedtuple
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import g2o

from utils.utils import get_test_points_pixel_and_world_coords, visualize_depth_map, perturb_transformation_matrix, \
                        get_point_cloud_from_pixels, visualize_matching_point_clouds, \
                        invert_transformation_matrix, is_valid_rotation_matrix, \
                        reprojection_error_test
from utils.optimize import LocalBA
from utils.metrics import RTE, RRE, ATE_rot, ATE_trans

if __name__ == '__main__':
    random.seed(5)
    np.random.seed(5)
    path = '/home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted'
    scene = 'scene0655_01'
    split = 'valid'
    test_id = 7

    ids = [(file.split('.')[0]) for file in os.listdir(path + "/" + split + "/rgb/" + scene)]
    ids = sorted(ids, key=lambda x: str(x))

    rgb_timestamps = [float(id) for id in ids]
    depth_timestamps = rgb_timestamps
    rgb_ids = [path + '/' + split + '/rgb/' + scene + "/" + str(item) + '.jpg' for item in ids]
    depth_ids = [path + '/' + split + '/depth/' + scene + "/" + str(item) + '.png' for item in ids]

    # Pick test images #
    img_test_id_str = ids[test_id]
    img_test_path = rgb_ids[test_id]
    depth_test_path = depth_ids[test_id]

    img_base_id_str = ids[0]
    img_base_path = ids[0]

    # Intrinsics #
    with open(path + "/" + split + '/rgb_intrinsics/' + scene + ".json", 'r') as json_file:
        data = json.load(json_file)
        cam = namedtuple('camera', 'fx fy cx cy scale')(data['fx'], data['fy'], data['cx'], data['cy'], 1000.0)

    # Extrinsics test image #
    with open(path + "/" + split + '/extrinsics/' + scene + "/" + img_test_id_str + ".json", 'r') as json_file:
        data = json.load(json_file)
        qx_test, qy_test, qz_test, qw_test = data['quat_x'], data['quat_y'], data['quat_z'], data['quat_w']
        x_test, y_test, z_test = data['x'], data['y'], data['z']

    # From camera to world #
    rotation_test = R.from_quat([qx_test, qy_test, qz_test, qw_test])
    transformation_matrix_test = np.eye(4)
    transformation_matrix_test[:3, :3] = rotation_test.as_matrix()
    transformation_matrix_test[:3, 3] = [x_test, y_test, z_test]
    assert is_valid_rotation_matrix(transformation_matrix_test[:3, :3])

    # Extrinsics base image #
    with open(path + "/" + split + '/extrinsics/' + scene + "/" + img_base_path + ".json", 'r') as json_file:
        data = json.load(json_file)
        qx_base, qy_base, qz_base, qw_base = data['quat_x'], data['quat_y'], data['quat_z'], data['quat_w']
        x_base, y_base, z_base = data['x'], data['y'], data['z']

    # From camera to world: first frame #
    rotation_base = R.from_quat([qx_base, qy_base, qz_base, qw_base])
    transformation_matrix_base = np.eye(4)
    transformation_matrix_base[:3, :3] = rotation_base.as_matrix()
    transformation_matrix_base[:3, 3] = [x_base, y_base, z_base]
    transformation_matrix_base_inv = invert_transformation_matrix(transformation_matrix_base)
    assert is_valid_rotation_matrix(transformation_matrix_base[:3, :3])

    # Base frame is not at zero, transform frames to the origin #
    # camera_to_world_transform for test frame                  #
    camera_to_world_transform = transformation_matrix_base_inv @ transformation_matrix_test
    assert is_valid_rotation_matrix(camera_to_world_transform[:3, :3])

    # Load test image and depth #
    img_test = cv2.imread(img_test_path, -1)
    depth_test = cv2.imread(depth_test_path, -1)
    depth_test = np.asarray(depth_test)
    depth_test = depth_test / cam.scale

    #visualize_depth_map(depth_test)

    # Get some 3D points #
    pixels, points_w = get_test_points_pixel_and_world_coords(cam=cam,
                                                              camera_to_world_transform=camera_to_world_transform,
                                                              image=img_test, depth_map=depth_test, num_points=50,
                                                              viz=False)

    world_to_camera_transform = invert_transformation_matrix(camera_to_world_transform)

    reprojection_error = reprojection_error_test(pixels[0, 5], pixels[1, 5], cam, points_w[5,:], \
                                                 world_to_camera_transform)
    assert np.all(np.allclose(reprojection_error, 0, atol=0.0001))

    # Perturb pose #
    translation_perturbation = np.array([0.05, 0.05, 0.05])  # cm
    rotation_perturbation = np.array([3, 3, 3])  # degrees
    camera_to_world_transform_perturb = perturb_transformation_matrix(camera_to_world_transform,
                                                                      translation_perturbation, rotation_perturbation)
    assert is_valid_rotation_matrix(camera_to_world_transform_perturb[:3, :3])

    # Get perturb points #
    perturbation_level = 0.05
    noise = np.random.normal(scale=perturbation_level, size=points_w.shape)
    points_w_perturb = points_w + noise
    #visualize_matching_point_clouds(points_w, points_w_perturb)

    #####################
    # Test optimization #
    #####################

    # Assume scale = 1 and canonical_depth == depth camera #
    canonical_depth = depth_test[pixels[1, :], pixels[0, :]]

    w_pose_c = g2o.SE3Quat(camera_to_world_transform_perturb[:3, :3], camera_to_world_transform_perturb[:3, 3])

    optimizer = LocalBA()
    optimizer.set_data(w_pose_c, cam, points_w_perturb, pixels, canonical_depth, scale_network=1, scale=1)
    optimizer.optimize(3)
    #print(optimizer.get_bad_measurements())

    ###############
    # Get results #
    ###############

    estimated_transformation = optimizer.get_poses()[0].matrix()
    estimated_scale = optimizer.get_scales()[0]

    print("Estimated scaale: ", estimated_scale)

    ####################
    # Calculate errors #
    ####################
    print("Scale-Aware test")

    print("Before optimization: ")
    rte = RTE(camera_to_world_transform, camera_to_world_transform_perturb)
    rre = RRE(camera_to_world_transform, camera_to_world_transform_perturb)
    ate_rot = ATE_rot(camera_to_world_transform, camera_to_world_transform_perturb)
    ate_trans = ATE_trans(camera_to_world_transform, camera_to_world_transform_perturb)
    print("relative trans error: ", rte)
    print("relative rot error: ", rre)
    print("absolute traj error (rot): ", ate_rot)
    print("absolute traj error (trans): ", ate_trans)

    print("After optimization: ")
    rte = RTE(camera_to_world_transform, estimated_transformation)
    rre = RRE(camera_to_world_transform, estimated_transformation)
    ate_rot = ATE_rot(camera_to_world_transform, estimated_transformation)
    ate_trans = ATE_trans(camera_to_world_transform, estimated_transformation)
    print("relative trans error: ", rte)
    print("relative rot error: ", rre)
    print("absolute traj error (rot): ", ate_rot)
    print("absolute traj error (trans): ", ate_trans)
