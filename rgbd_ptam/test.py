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
from optimization import BundleAdjustmentScaleAware



def optimize(args, frame_1, frame_2, matched_measurements_1, matched_measurements_2, class_id=0, max_iterations=10, gt_pose=None):
    optimizer = BundleAdjustmentScaleAware(args=args)
    optimizer.clear()

    # Poses #
    optimizer.add_pose(0, frame_1.pose, frame_1.cam, fixed=True)
    optimizer.add_pose(1, frame_2.pose, frame_2.cam, fixed=False)
    
    # Scales #
    optimizer.add_scale(2, frame_1.scale_aware_frame.scales[class_id], fixed=True)
    optimizer.add_scale(3, frame_1.scale_aware_frame.scales[class_id], fixed=True)


    # Scales edges network #
    information_1 = np.identity(1)
    information_2 = np.identity(1)
    if args.use_uncertainties:
        information_1 *= 1. / frame_1.scale_aware_frame.scales_uncertainty[class_id]
        information_2 *= 1. / frame_2.scale_aware_frame.scales_uncertainty[class_id]
    optimizer.add_scale_edge(4, scale_id=2, meas=frame_1.scale_aware_frame.scales[class_id],
                                  information=information_1)
    optimizer.add_scale_edge(5, scale_id=3, meas=frame_2.scale_aware_frame.scales[class_id],
                                  information=information_2)


    index_point = 0
    index_general = 0
    points_start = 6
    depth_scale_consistenct_start = 10000
    camera_start = 50000

    # Points frame 1 #
    for meas in matched_measurements_1:  
        pt = meas.mappoint

        xy = meas.xy 

        # 3D Landmark #
        optimizer.add_point(index_point + points_start, pt.position, fixed=True)
        
        optimizer.add_camera_edge(index_general + camera_start, point_id=index_point + points_start, pose_id=0, meas=meas.xy)

        
        information = np.identity(1)
        if args.use_uncertainties:
            information *= 1. / m.covariance_canonical_measurement
        optimizer.add_depth_scale_consistency_edge(index_general + depth_scale_consistenct_start, point_id=index_point + points_start, pose_id=0, \
                                                   scale_id=2, meas=meas.canonical_measurement, information=information)
        index_general += 1
        index_point += 1
    
    # Frame 2 # 
    index_point = 0
    for meas in matched_measurements_2:  
        pt = meas.mappoint

        optimizer.add_camera_edge(index_general + camera_start, point_id=index_point + points_start, pose_id=1, meas=meas.xy)

        information = np.identity(1)
        if args.use_uncertainties:
            information *= 1. / m.covariance_canonical_measurement
        optimizer.add_depth_scale_consistency_edge(index_general + depth_scale_consistenct_start, point_id=index_point + points_start, pose_id=1, \
                                                   scale_id=3, meas=meas.canonical_measurement, information=information)
        index_general += 1
        index_point += 1
     

    optimizer.optimize(max_iterations)


def read_args(parser):
    parser.add_argument('--no-viz', action='store_true', help='do not visualize')
    parser.add_argument('--debug', action='store_true', help='debug slam')
    parser.add_argument('--dataset', type=str, help='dataset (TUM/ICL-NUIM)', default='scannet')
    parser.add_argument('--path', type=str, help='dataset path',
                        default='usr/stud/petp/storage/user/petp/datasets/scannet/data_converted')
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
    parser.add_argument('--weight_camera', type=float, default=2, help='Weight for camera loss') # 1
    parser.add_argument('--threshold_depth_consistency', type=float, default=0.0025,
                        help='Threshold for huber loss depth consistency')
    parser.add_argument('--weight_depth_consistency', type=float, default=1, help='Weight for depth consistency loss') # 0.5
    parser.add_argument('--threshold_scale', type=float, default=0.0025, help='Threshold for huber loss scale')
    parser.add_argument('--weight_scale', type=float, default=2, help='Weight for scale loss') # 0.5
    parser.add_argument('--scene', type=str, default='scene0655_01')
    parser.add_argument('--split', type=str, default='valid')

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
        feature.filter_features_to_class(sem_mask, 0, draw=False)
           
    frame = RGBDFrame(idx, g2o.Isometry3d(), feature, depth, cam, timestamp=timestamp,
                      canonical=dataset.canonical[idx], canonical_uncertainty=dataset.canonical_uncertainty[idx],
                      scales=scales, scales_uncertainty=scales_uncertainty,
                      pixel_to_scale_map=dataset.pixel_to_scale_map[idx], scales_valid=scales_valid)

    return frame, gt_pose, sem_mask

def normalize_angle(angle):
    return (angle + 180) % 360 - 180

def add_noise(pose, rot_noise=2, trans_noise=0.1):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = read_args(parser)

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
        
    # Frames #
    frame_1, pose_w_c1, sem_mask_1 = get_frame(0)
    frame_2, pose_w_c2, sem_mask_2 = get_frame(3, mask=True, class_id=0)

    pose_c1_c2 = pose_w_c1.inverse() * pose_w_c2
    pose_c1_c2_noise = add_noise(pose_c1_c2)
    frame_2.update_pose(pose_c1_c2_noise)
    
    frame_1 = frame_1.to_keyframe()
    frame_2 = frame_2.to_keyframe()

    mappoints_1, measurements_1 = frame_1.cloudify()  

    matched_measurements_2, matched_ids = frame_2.match_mappoints_and_get_ids(mappoints_1, Measurement.Source.TRACKING)
    matched_measurements_1 = np.asarray(measurements_1)[matched_ids]
   
    optimize(args, frame_1, frame_2, matched_measurements_1, matched_measurements_2, class_id=0, max_iterations=10, gt_pose=pose_c1_c2)
