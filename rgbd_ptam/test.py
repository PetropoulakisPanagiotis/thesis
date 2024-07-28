import time
import argparse
import g2o
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from dataset import ScanNetDataset
from components import Camera, RGBDFrame, Measurement
from params import Params
from feature import ImageFeature
from covisibility import CovisibilityGraph
from utils_tests import draw_keypoints, test_1, test_2


def read_args(parser):
    parser.add_argument('--no-viz', action='store_true', help='do not visualize')
    parser.add_argument('--debug', action='store_true', help='debug slam')
    parser.add_argument('--dataset', type=str, help='dataset (TUM/ICL-NUIM)', default='scannet')
    parser.add_argument('--path', type=str, help='dataset path',
                        default='/usr/stud/petp/storage/user/petp/datasets/scannet/data_converted')
    parser.add_argument('--scale_aware', action='store_true', default=False, help='Scale-Aware slam loss enable')
    parser.add_argument('--network_depth', action='store_true', default=False,
                        help='Use ground truth depth or network depth')
    parser.add_argument('--optimization_base_type', type=str, default='mono', choices=['mono', 'virtual'],
                        help='Monocular or virtual stereo')
    parser.add_argument('--use_uncertainties', action='store_true', default=False,
                        help='use uncertainties during optimization')
    parser.add_argument('--optimization_type', type=str, default='global',
                        choices=['global', 'per_class', 'per_instance'], help='Scale-Aware variation')
    parser.add_argument('--out_path', type=str, default='./results_debug', help='Folder to save results')
    parser.add_argument('--total', type=int, default=None, help='Total number of frame')
    parser.add_argument('--exp_name', type=str, default='exp_1', help='Experiment name')
    parser.add_argument('--threshold_camera', type=float, default=5.991, help='Threshold for huber loss camera')
    parser.add_argument('--weight_camera', type=float, default=0.1, help='Weight for camera loss') # 1
    parser.add_argument('--threshold_depth_consistency', type=float, default=0.02,
                        help='Threshold for huber loss depth consistency')
    parser.add_argument('--weight_depth_consistency', type=float, default=1, help='Weight for depth consistency loss') # 0.5
    parser.add_argument('--threshold_scale', type=float, default=0.02, help='Threshold for huber loss scale')
    parser.add_argument('--weight_scale', type=float, default=0, help='Weight for scale loss') # 0.5
    parser.add_argument('--scene', type=str, default='scene0655_01')
    parser.add_argument('--split', type=str, default='valid')
    parser.add_argument('--test_case', type=int, default=1, help='Test case')

    args = parser.parse_args()
    return args


def get_frame(idx, mask=False, class_id=0):
    time_start = time.time()
    timestamp = dataset.timestamps[idx]
        
    depth = dataset.depth[idx]

    scales, scales_uncertainty, _, scales_valid = dataset.scale[idx]

    gt_pose = dataset.gt_poses[idx]    
    sem_mask = dataset.sem_masks[idx]
    
    feature = ImageFeature(dataset.rgb[idx], params)
    feature.extract()  # Detect keypoints and descriptors of image
    if mask:
        #feature.filter_features_to_class(sem_mask, 0, draw=True)
        feature.filter_features_to_class(sem_mask, 0, draw=False)
    frame = RGBDFrame(idx, g2o.Isometry3d(), feature, depth, cam, timestamp=timestamp,
                      canonical=dataset.canonical[idx], canonical_uncertainty=dataset.canonical_uncertainty[idx],
                      scales=scales, scales_uncertainty=scales_uncertainty,
                      pixel_to_scale_map=dataset.pixel_to_scale_map[idx], scales_valid=scales_valid)

    return frame, gt_pose, sem_mask


def normalize_angle(angle):
    return (angle + 180) % 360 - 180


def add_noise(pose, rot_noise=5, trans_noise=0.5):
    transformation = pose.matrix()
    rot = transformation[:3, :3]
    trans = transformation[:3, 3]

    rot = R.from_matrix(rot)
    euler = rot.as_euler('xyz', degrees=True)
    euler += rot_noise
    euler = normalize_angle(euler)
    
    trans += trans_noise 
    rot = R.from_euler('xyz', euler)
    quat = rot.as_quat()
    noisy_pose = g2o.Isometry3d(g2o.Quaternion(quat), trans)
    return noisy_pose


def add_noise_points(measurements, x=0.05, y=0.05, z=0.05):
    for meas in measurements:
        meas.mappoint.position = meas.mappoint.position + np.asarray([x, y, z])

    return measurements

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = read_args(parser)
    test_case = args.test_case
    params = Params()
    
    # Per-Class SLAM #
    args.scale_aware = True
    args.network_depth = True
    args.optimization_type = 'per_class'
    args.scene = 'scene0574_01'
    print(f'[scene name {args.scene}]')

    print('[running per-class SLAM]')
    dataset = ScanNetDataset(args.path, args.scene, args.split, args.scale_aware, args.optimization_type,
                                 network_depth=args.network_depth, total=args.total)

    height, width = dataset.rgb.shape[:2]
    cam = Camera(dataset.cam.fx, dataset.cam.fy, dataset.cam.cx, dataset.cam.cy, width, height, dataset.cam.scale,
                 params.virtual_baseline, params.depth_near, params.depth_far, params.frustum_near, params.frustum_far)
        
    # Noise to Pose #
    if test_case == 1:
        print("test-case: camera-reprojection - noise in pose\n")
        
        # Frames #
        frame_1, pose_w_c1, sem_mask_1 = get_frame(0)
        frame_1 = frame_1.to_keyframe()

        noise_pose = add_noise(g2o.Isometry3d(), rot_noise=5, trans_noise=0.5)
        frame_1.update_pose(noise_pose) 
        mappoints_1, measurements_1 = frame_1.cloudify() 
        
        frame_1.update_pose(g2o.Isometry3d()) 
        test_1(args, frame_1, measurements_1, max_iterations=25, gt_pose=noise_pose)
    
    # Noise to Points and Pose #
    elif test_case == 2:
        print("test-case: camera-reprojection - noise in pose and points\n")

        # Frames #
        frame_1, pose_w_c1, sem_mask_1 = get_frame(0)
        frame_1 = frame_1.to_keyframe()

        noise_pose = add_noise(g2o.Isometry3d(), rot_noise=5, trans_noise=0.5)
        frame_1.update_pose(noise_pose) 
        mappoints_1, measurements_1 = frame_1.cloudify() 
        measurements_1 = add_noise_points(measurements_1, x=0.02, y=0.02, z=0.02) 

        frame_1.update_pose(g2o.Isometry3d()) 
        test_1(args, frame_1, measurements_1, max_iterations=30, gt_pose=noise_pose)
        # Noise to Points and Pose #
    elif test_case == 3:
        print("test-case: camera-reprojection - semantic class 0 - noise in pose and points\n")

        # Frames #
        frame_1, pose_w_c1, sem_mask_1 = get_frame(0, mask=True, class_id=0)
        frame_1 = frame_1.to_keyframe()

        noise_pose = add_noise(g2o.Isometry3d(), rot_noise=5, trans_noise=0.5)
        frame_1.update_pose(noise_pose) 
        mappoints_1, measurements_1 = frame_1.cloudify() 
        measurements_1 = add_noise_points(measurements_1, x=0.02, y=0.02, z=0.02) 

        frame_1.update_pose(g2o.Isometry3d()) 
        test_1(args, frame_1, measurements_1, max_iterations=50, gt_pose=noise_pose)
    elif test_case == 4:
        print("test-case: scale-aware - semantic class 0 - noise in pose and points and scale\n")
        
        # Frames #
        frame_1, pose_w_c1, sem_mask_1 = get_frame(0, mask=True, class_id=0)
        frame_1 = frame_1.to_keyframe()

        noise_pose = add_noise(g2o.Isometry3d(), rot_noise=5, trans_noise=0.5)
        frame_1.update_pose(noise_pose) 
        mappoints_1, measurements_1 = frame_1.cloudify() 
        measurements_1 = add_noise_points(measurements_1, x=0.02, y=0.02, z=0.02) 

        frame_1.scale_aware_frame.scales[0] = 3 # 4.7979

        frame_1.update_pose(g2o.Isometry3d()) 
        test_2(args, frame_1, measurements_1, class_id=0, max_iterations=500, gt_pose=noise_pose)
    else:        
        exit()


    """    
    pose_c1_c2 = pose_w_c1.inverse() * pose_w_c2
    # c1 -> w * w -> c2 == c1 -> c2 #
    pose_c2_c1 = pose_c1_c2.inverse()
    pose_c2_c1_noise = add_noise(pose_c2_c1)
        # Frames #
        frame_1, pose_w_c1, sem_mask_1 = get_frame(0)#, mask=True, class_id=0)
        frame_2, pose_w_c2, sem_mask_2 = get_frame(2, mask=True, class_id=0)

        frame_1 = frame_1.to_keyframe()
        frame_2 = frame_2.to_keyframe()


    #`frame_2.update_pose(pose_c2_c1) # To find good correspondances 
    matched_measurements_2, matched_ids = frame_2.match_mappoints_and_get_ids(mappoints_1, Measurement.Source.TRACKING)
    matched_measurements_1 = np.asarray(measurements_1)[matched_ids]
    assert len(matched_measurements_1) == len(matched_measurements_2) 

    keypoints_1 = [[cv2.KeyPoint(x=int(meas.xy[0]), y=int(meas.xy[1]), size=1) for meas in matched_measurements_1][15]]    
    #draw_keypoints(frame_1.rgb.image, keypoints_1)
    keypoints_2 = [[cv2.KeyPoint(x=int(meas.xy[0]), y=int(meas.xy[1]), size=1) for meas in matched_measurements_2][15]]
    #draw_keypoints(frame_2.rgb.image, keypoints_2)
    
    #frame_2.update_pose(pose_c2_c1_noise)
    #optimize(args, frame_1, frame_2, matched_measurements_1, matched_measurements_2, class_id=0, max_iterations=0, gt_pose=pose_c2_c1)
    
    #optimize(args, frame_1, frame_2, matched_measurements_1, matched_measurements_2, class_id=0, max_iterations=0, gt_pose=frame_1.pose)
    """
