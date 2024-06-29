import numpy as np
from scipy.spatial.transform import Rotation

import json
import time
from itertools import chain
from collections import defaultdict

from covisibility import CovisibilityGraph
from optimization import BundleAdjustment, BundleAdjustmentScaleAware
from mapping import Mapping
from mapping import MappingThread
from components import Measurement
from motion import MotionModel
from loopclosing import LoopClosing
from evaluate import read_trajectory, evaluate_trajectory
import associate
from evaluate_ate import align

import pandas as pd
"""
Three threads:
              - mapping
              - loopclosure
              - main -> tracking
              - TODO: another thread for fullBA
Two processed:
              - main
              - viewer
"""


class Tracking(object):
    def __init__(self, params, args):
        self.params = params
        self.args = args
        self.optimizer = BundleAdjustment(args) if not args.scale_aware else BundleAdjustmentScaleAware(args=args)
        self.min_measurements = params.pnp_min_measurements
        self.max_iterations = params.pnp_max_iterations

    def refine_pose(self, pose, cam, measurements, scale_aware_frame=None):
        assert len(measurements) >= self.min_measurements, ('Not enough points')

        self.optimizer.clear()

        if self.args.scale_aware:
            # Pose Id: scale_offset + 2*id
            self.optimizer.add_pose(0, pose, cam, fixed=False)

            # Scales: edge and vertex #
            for ii, (scale, scale_valid) in enumerate(zip(scale_aware_frame.scales, scale_aware_frame.scales_valid)):
                if scale_valid == 1:
                    # Scale Id: ii
                    self.optimizer.add_scale(ii, scale, fixed=True)

            # BA with one pose #
            for ii, m in enumerate(measurements):
                # Point Id: scale_offset + 2*ii + 1
                self.optimizer.add_point(ii, m.mappoint.position, fixed=True)
                # Edge Id: scale_offset + 2*ii
                self.optimizer.add_camera_edge(ii, ii, 0, m.xy)

                information = np.identity(1)
                if self.args.use_uncertainties:
                    information *= 1. / m.covariance_canonical_measurement

                # Edge Id: scale_offset + 2*ii + 1
                self.optimizer.add_depth_scale_consistency_edge(ii, ii, 0, m.scale_id_measurement,
                                                                m.canonical_measurement, information=information)
        else:
            self.optimizer.add_pose(0, pose, cam, fixed=False)
            # BA with one pose #
            for i, m in enumerate(measurements):
                self.optimizer.add_point(i, m.mappoint.position, fixed=True)
                self.optimizer.add_edge(0, i, 0, m)
        self.optimizer.optimize(self.max_iterations)
        return self.optimizer.get_pose(0)


class RGBDPTAM(object):
    def __init__(self, params, args):
        self.params = params
        self.args = args

        self.graph = CovisibilityGraph()
        self.mapping = MappingThread(self.graph, params, args)
        self.tracker = Tracking(params, args)
        self.motion_model = MotionModel(params)

        self.loop_closing = None  #LoopClosing(self, params)
        self.loop_correction = None

        self.reference = None  # reference keyframe
        self.preceding = None  # last keyframe
        self.current = None  # current frame
        self.candidates = []  # candidate keyframes
        self.results = []  # tracking results bool

        self.status = defaultdict(bool)
        self.non_keyframes = []

    def stop(self):
        self.mapping.stop()
        if self.loop_closing is not None:
            self.loop_closing.stop()

    def initialize(self, frame):
        mappoints, measurements = frame.cloudify()
        assert len(mappoints) >= self.params.init_min_points, ('Not enough points to initialize map.')

        keyframe = frame.to_keyframe()
        keyframe.set_fixed(True)
        self.graph.add_keyframe(keyframe)
        self.mapping.add_measurements(keyframe, mappoints, measurements)
        if self.loop_closing is not None:
            self.loop_closing.add_keyframe(keyframe)

        self.reference = keyframe
        self.preceding = keyframe
        self.current = keyframe
        self.status['initialized'] = True

        self.motion_model.update_pose(frame.timestamp, frame.position, frame.orientation)

    def track(self, frame):
        tracking_fail = False
        while self.is_paused():
            time.sleep(1e-4)
        self.set_tracking(True)

        self.current = frame
        if not args.no_viz:
            print('Tracking:', frame.idx, ' <- ', self.reference.id, self.reference.idx)

        # set pose to frame from tracking
        predicted_pose, _ = self.motion_model.predict_pose(frame.timestamp)
        frame.update_pose(predicted_pose)
        if self.loop_closing is not None:
            if self.loop_correction is not None:
                estimated_pose = g2o.Isometry3d(frame.orientation, frame.position)
                estimated_pose = estimated_pose * self.loop_correction
                frame.update_pose(estimated_pose)
                self.motion_model.apply_correction(self.loop_correction)
                self.loop_correction = None

        # Get local map and also find measurements for current frame #
        local_mappoints = self.filter_points(frame)
        measurements = frame.match_mappoints(local_mappoints, Measurement.Source.TRACKING)

        # Measurements -> frame
        # Local        -> local map
        if not args.no_viz:
            print('measurements:', len(measurements), '   ', len(local_mappoints))

        # Update featrues                      #
        # tracked_map -> measurements of frame #
        tracked_map = set()
        for m in measurements:
            mappoint = m.mappoint
            mappoint.update_descriptor(m.get_descriptor())
            mappoint.increase_measurement_count()
            tracked_map.add(mappoint)

        try:
            # Find most probable reference frame #
            self.reference = self.graph.get_reference_frame(tracked_map)

            # BA only 1 pose #
            pose = self.tracker.refine_pose(frame.pose, frame.cam, measurements, frame.scale_aware_frame)

            frame.update_pose(pose)
            self.motion_model.update_pose(frame.timestamp, pose.position(), pose.orientation())

            # tracking succeed
            self.candidates.append(frame)
            self.params.relax_tracking(False)
            self.results.append(True)
        except:
            # tracking failed - likely not enough points
            self.params.relax_tracking(True)
            self.results.append(False)
            tracking_fail = True

        remedy = False
        if self.results[-2:].count(False) == 2:  # Two last frames failed
            if len(self.candidates
                   ) > 0 and self.candidates[-1].idx > self.preceding.idx:  # Remedy is not yet a keyframe
                frame = self.candidates[-1]  # Get last valid tracked frame
                remedy = True
            else:
                tracking_fail = True
                print('tracking failed!')

        # Remedy frame used or tracking succeded and should be keyframe #
        if remedy or (self.results[-1] and self.should_be_keyframe(frame, measurements)):  # or (self.results[-1]):
            keyframe = frame.to_keyframe()
            keyframe.update_reference(self.reference)
            keyframe.update_preceding(self.preceding)

            self.mapping.add_keyframe(keyframe, measurements)
            if self.loop_closing is not None:
                self.loop_closing.add_keyframe(keyframe)

            self.preceding = keyframe
            if not args.no_viz:
                print('new keyframe', keyframe.idx)
        else:  # Remedy already used as keyframe
            if tracking_fail is not True:
                self.non_keyframes.append(frame)

        self.set_tracking(False)

    def filter_points(self, frame):
        # Get local 3D points #
        # local_mappoints = self.graph.get_local_map(self.tracked_map)[0]
        local_mappoints = self.graph.get_local_map_v2([self.preceding, self.reference])[0]

        # If we can project the 3D map points to the current frame #
        can_view = frame.can_view(local_mappoints)
        if not args.no_viz:
            print('filter points:', len(local_mappoints), can_view.sum(), len(self.preceding.mappoints()),
                  len(self.reference.mappoints()))

        checked = set()
        filtered = []
        for i in np.where(can_view)[0]:
            pt = local_mappoints[i]
            if pt.is_bad():
                continue
            pt.increase_projection_count()
            filtered.append(pt)
            checked.add(pt)

        # Also add points from preceding and reference #
        for reference in set([self.preceding, self.reference]):
            for pt in reference.mappoints():  # neglect can_view test
                if pt in checked or pt.is_bad():
                    continue
                pt.increase_projection_count()
                filtered.append(pt)

        return filtered

    def should_be_keyframe(self, frame, measurements):
        if self.adding_keyframes_stopped():
            return False

        n_matches = len(measurements)
        return (n_matches < self.params.min_tracked_points and n_matches > self.params.pnp_min_measurements)

    def set_loop_correction(self, T):
        self.loop_correction = T

    def is_initialized(self):
        return self.status['initialized']

    def pause(self):
        self.status['paused'] = True

    def unpause(self):
        self.status['paused'] = False

    def is_paused(self):
        return self.status['paused']

    def is_tracking(self):
        return self.status['tracking']

    def set_tracking(self, status):
        self.status['tracking'] = status

    def stop_adding_keyframes(self):
        self.status['adding_keyframes_stopped'] = True

    def resume_adding_keyframes(self):
        self.status['adding_keyframes_stopped'] = False

    def adding_keyframes_stopped(self):
        return self.status['adding_keyframes_stopped']

    def save_results(self):
        self.non_keyframes.extend(self.mapping.graph.keyframes())
        all_frames = self.non_keyframes
        #all_frames = self.mapping.graph.keyframes()
        sorted_frames = sorted(all_frames, key=lambda obj: obj.timestamp)
        with open(args.result_path + 'slam.txt', 'w') as file_result:
            for kf in sorted_frames:
                timestamp = kf.timestamp  #1. (room),
                matrix = kf.pose.matrix()[:3]
                xyz = matrix[:3, 3]
                r = Rotation.from_matrix(matrix[:3, :3])
                q = r.as_quat().tolist()

                current_result = [timestamp]
                current_result.extend(xyz)
                current_result.extend(q)
                current_result = [str(item) for item in current_result]
                line = ' '.join(current_result)
                file_result.write(line + '\n')

        # Save optimized scale #
        if self.args.scale_aware:
            total_mappoints = 0
            for keyframe in self.mapping.graph.keyframes():
                scale_data = {
                    'scale': keyframe.scale_aware_frame.scales,
                    'scale_type': args.optimization_type,
                    'scale_uncertainty': keyframe.scale_aware_frame.scales_uncertainty
                }
                path = args.scale_path + str(
                    f'{int(keyframe.rgb.timestamp):05}') + '.json'
                with open(path, 'w') as file:
                    json.dump(scale_data, file, indent=4)

                path_scale_map = args.scale_path + str(
                    f'{int(keyframe.rgb.timestamp):05}') + '_scale_map.png'
                scale_map = np.zeros((keyframe.rgb.image.shape[:2]))
                #depth_map = np.zeros((keyframe.rgb.image.shape[:2]))
                #path_depth_map = args.out_path + args.optimization_type + '/optimized_scale/' + str(f'{int(keyframe.rgb.timestamp):05}') + '_depth_map.png'

                total_mappoints += len(keyframe.measurements())
                for m in keyframe.measurements():
                    xy = [int(item) for item in m.xy]
                    scale_map[xy[1], xy[0]] = 1
                    #depth_map[xy[1], xy[0]] = keyframe.transform(m.mappoint.position)[2] / 1000

                cv2.imwrite(path_scale_map, scale_map, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                #cv2.imwrite(path_depth_map, depth_map, [cv2.IMWRITE_PNG_COMPRESSION, 9])

        print("results saved...")


def main_loop(args):
    if 'tum' in args.dataset.lower():
        dataset = TUMRGBDDataset(args.path)
    elif 'icl' in args.dataset.lower():
        dataset = ICLNUIMDataset(args.path)
    elif 'scannet' in args.dataset.lower():
        dataset = ScanNetDataset(args.path, args.scene, args.split, args.scale_aware, args.optimization_type,
                                 network_depth=args.network_depth, total=args.total)

    params = Params()
    height, width = dataset.rgb.shape[:2]
    cam = Camera(dataset.cam.fx, dataset.cam.fy, dataset.cam.cx, dataset.cam.cy, width, height, dataset.cam.scale,
                 params.virtual_baseline, params.depth_near, params.depth_far, params.frustum_near, params.frustum_far)
    ptam = RGBDPTAM(params, args)

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    if args.scale_aware:
        if not os.path.exists(args.scale_path):
            os.makedirs(args.scale_path)

    if not args.no_viz:
        from viewer import MapViewer
        viewer = MapViewer(ptam, params)

    durations = []
    for i in range(len(dataset))[:]:
        feature = ImageFeature(dataset.rgb[i], params)
        depth = dataset.depth[i]
        if dataset.timestamps is None:
            timestamp = i / 20.
        else:
            timestamp = dataset.timestamps[i]
        time_start = time.time()
        feature.extract()  # Detect keypoints and descriptors of image
        if args.scale_aware:
            scales, scales_uncertainty, _, scales_valid = dataset.scale[i]
            frame = RGBDFrame(i, g2o.Isometry3d(), feature, depth, cam, timestamp=timestamp,
                              canonical=dataset.canonical[i], canonical_uncertainty=dataset.canonical_uncertainty[i],
                              scales=scales, scales_uncertainty=scales_uncertainty,
                              pixel_to_scale_map=dataset.pixel_to_scale_map[i], scales_valid=scales_valid)
        else:
            frame = RGBDFrame(i, g2o.Isometry3d(), feature, depth, cam, timestamp=timestamp)

        if not ptam.is_initialized():
            ptam.initialize(frame)
        else:
            ptam.track(frame)

        duration = time.time() - time_start
        durations.append(duration)
        if not args.no_viz:
            print('duration', duration)
            print()

        if not args.no_viz:
            viewer.update()

    print('num frames', len(durations))
    print('num keyframes', len(ptam.graph.keyframes()))
    print(f'average time {np.mean(durations):.3f}')

    print("saving results...")
    ptam.save_results()

    ptam.mapping.create_map(args.result_path + 'map.pcd')

    ptam.stop()
    if not args.no_viz:
        viewer.stop()

    durations_dict = {'mean': np.mean(durations), 'std': np.std(durations), 'max': np.max(durations), 'min': np.min(durations)}

    return args.result_path + 'slam.txt', durations_dict


def evaluate(slam_path, gt_path, max_difference=0.02):
    print("evaluating...")

    # Relative pose first #
    traj_gt = read_trajectory(gt_path)
    traj_est = read_trajectory(slam_path)

    result_eval = evaluate_trajectory(traj_gt, traj_est, 0, False, 1.0, 's', 0.0, 1.0)

    trans_error = np.array(result_eval)[:, 4]
    rot_error = np.array(result_eval)[:, 5]

    result = {}

    result['pairs'] = len(trans_error)

    result['trans_rmse'] = np.sqrt(np.dot(trans_error, trans_error) / len(trans_error))
    result['trans_mean'] = np.mean(trans_error)
    result['trans_median'] = np.median(trans_error)
    result['trans_std'] = np.std(trans_error)
    result['trans_min'] = np.min(trans_error)
    result['trans_max'] = np.max(trans_error)

    result['rot_rmse'] = (np.sqrt(np.dot(rot_error, rot_error) / len(rot_error)) * 180.0 / np.pi)
    result['rot_mean'] = (np.mean(rot_error) * 180.0 / np.pi)
    result['rot_median'] = (np.median(rot_error) * 180.0 / np.pi)
    result['rot_std'] = (np.std(rot_error) * 180.0 / np.pi)
    result['rot_min'] = (np.min(rot_error) * 180.0 / np.pi)
    result['rot_max'] = (np.max(rot_error) * 180.0 / np.pi)

    # Absolute trajectory #
    first_list = associate.read_file_list(slam_path)
    second_list = associate.read_file_list(gt_path)

    matches = associate.associate(first_list, second_list, 0.0, max_difference)
    if len(matches) < 2:
        sys.exit(
            "Couldn't find matching timestamp pairs between groundtruth and estimated trajectory! Did you choose the correct sequence?"
        )

    first_xyz = np.matrix([[float(value) for value in first_list[a][0:3]] for a, b in matches]).transpose()
    second_xyz = np.matrix([[float(value) * float(1) for value in second_list[b][0:3]] for a, b in matches]).transpose()

    rot, trans, trans_error = align(second_xyz, first_xyz)

    result_ate = {}
    result_ate['pairs'] = len(trans_error)
    result_ate['abs_trans_rmse'] = np.sqrt(np.dot(trans_error, trans_error) / len(trans_error))
    result_ate['abs_trans_mean'] = np.mean(trans_error)
    result_ate['abs_trans_median'] = np.median(trans_error)
    result_ate['abs_trans_min'] = np.min(trans_error)
    result_ate['abs_trans_max'] = np.max(trans_error)

    print("evaluation done...")
    return result, result_ate


if __name__ == '__main__':
    import cv2
    import g2o

    import os
    import sys
    import argparse

    from threading import Thread

    from components import Camera
    from components import RGBDFrame
    from feature import ImageFeature
    from params import Params
    from dataset import TUMRGBDDataset, ICLNUIMDataset, ScanNetDataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-viz', action='store_true', help='do not visualize')
    parser.add_argument('--dataset', type=str, default='ICL', help='dataset (TUM/ICL-NUIM)')
    parser.add_argument('--path', type=str, help='dataset path',
                        default='path/to/your/ICL-NUIM_RGBD/living_room_traj3_frei_png')

    parser.add_argument('--scale_aware', action='store_true', default=False, help='Scale-Aware slam loss enable')
    parser.add_argument('--network_depth', action='store_true', default=False,
                        help='Use ground truth depth or network depth')
    parser.add_argument('--optimization_base_type', type=str, default='mono', choices=['mono', 'virtual'],
                        help='Monocular or virtual stereo')
    parser.add_argument('--use_uncertainties', action='store_true', default=False,
                        help='use uncertainties during optimization')
    parser.add_argument('--optimization_type', type=str, default='global',
                        choices=['global', 'per_class', 'per_instance'], help='Scale-Aware variation')
    parser.add_argument('--out_path', type=str, default='./results/', help='Folder to save results')
    parser.add_argument('--total', type=int, default=None, help='Total number of frame')

    parser.add_argument('--exp_name', type=str, default='exp_1', help='Experiment name')

    parser.add_argument('--threshold_camera', type=float, default=5.991, help='Threshold for huber loss camera')
    parser.add_argument('--weight_camera', type=float, default=1, help='Weight for camera loss')

    parser.add_argument('--threshold_depth_consistency', type=float, default=3.991,
                        help='Threshold for huber loss depth consistency')
    parser.add_argument('--weight_depth_consistency', type=float, default=0.5, help='Weight for depth consistency loss')

    parser.add_argument('--threshold_scale', type=float, default=5.991, help='Threshold for huber loss scale')
    parser.add_argument('--weight_scale', type=float, default=2, help='Weight for scale loss')

    parser.add_argument('--scene', type=str, default='scene0655_01')

    parser.add_argument('--split', type=str, default='valid')
    args = parser.parse_args()

    total_runs = 3
    args.no_viz = True

    scenes = ['scene0655_01']

    methods_names = ['mono-gt', 'mono', 'virtual-gt', 'virtual', 'global', 'per-class']

    initial_path = args.out_path
    for scene in scenes:
        print(f'[scene name {scene}]')
        args.scene = scene
        args.out_path = initial_path + '/' + 'scene/'

        gt_path = args.path + '/valid/gt_traj/' + args.scene + '/gt_traj.txt'

        relative_df = pd.DataFrame()
        ate_df = pd.DataFrame()
        durations_df = pd.DataFrame(columns=['mean', 'std', 'max', 'min'])

        # Monocular SLAM with gt depth #
        args.scale_aware = False
        args.network_depth = False
        args.optimization_base_type = 'mono'
        mono_relative_list = []
        mono_ate_list = []
        durations_list = []

        print('[running monocular SLAM with gt depth]')
        for i in range(total_runs):
            print('\n')
            print(f'[iteration {i+1}]')

            args.exp_name = args.optimization_base_type + '_gt/' + args.optimization_base_type + '_gt_' + str(i) + '/'
            args.result_path = args.out_path + args.exp_name
            slam_path, durations_dict = main_loop(args)

            durations_list.append(durations_dict)

            eval_path = args.out_path + args.exp_name + 'evaluation.txt'
            result_dict, result_ate_dict = evaluate(slam_path, gt_path)

            mono_relative_error_path = args.result_path + 'relative_error_' + str(i) + '.csv'
            result_df = pd.DataFrame.from_dict(result_dict, orient='index')

            mono_ate_error_path = args.result_path + 'ate_error_' + str(i) + '.csv'
            result_ate_df = pd.DataFrame.from_dict(result_ate_dict, orient='index')

            result_df.to_csv(mono_relative_error_path)
            result_ate_df.to_csv(mono_ate_error_path)

            mono_relative_list.append(result_df)
            mono_ate_list.append(result_ate_df)

        mono_relative_df = pd.concat(mono_relative_list, axis=1, ignore_index=True)
        mono_ate_df = pd.concat(mono_ate_list, axis=1, ignore_index=True)

        durations = defaultdict(list)
        for item in durations_list:
            for key, value in item.items():
                durations[key].append(value)
        durations_avg = {key: sum(vals) / len(vals) for key, vals in durations.items()}
        durations_avg_df = pd.DataFrame.from_dict(durations_avg, orient='index').T
        durations_df = pd.concat([durations_df, durations_avg_df], axis=0, ignore_index=True)

        # Mean #
        mono_relative_means_df = mono_relative_df.mean(axis=1).to_frame().T
        mono_ate_means_df = mono_ate_df.mean(axis=1).to_frame().T

        mono_relative_error_mean_path = args.out_path + args.optimization_base_type + '_gt/' + '/relative_error.csv'
        mono_ate_error_mean_path = args.out_path + args.optimization_base_type + '_gt' + '/ate_error.csv'
        mono_relative_means_df.to_csv(mono_relative_error_mean_path)
        mono_ate_means_df.to_csv(mono_ate_error_mean_path)

        print(durations_avg_df)
        print(mono_ate_means_df)
        print(mono_relative_means_df)

        relative_df = pd.concat([mono_relative_means_df, relative_df], axis=0, ignore_index=True)
        ate_df = pd.concat([mono_ate_means_df, ate_df], axis=0, ignore_index=True)
        print('-----------------------------------------\n')

        # Monocular SLAM #
        args.scale_aware = False
        args.network_depth = True
        args.optimization_base_type = 'mono'
        mono_relative_list = []
        mono_ate_list = []
        durations_list = []
        print('[running monocular SLAM]')
        for i in range(total_runs):
            print('\n')
            print(f'[iteration {i+1}]')

            args.exp_name = args.optimization_base_type + '/' + args.optimization_base_type + '_' + str(i) + '/'
            args.result_path = args.out_path + args.exp_name
            slam_path, durations_dict = main_loop(args)

            durations_list.append(durations_dict)

            eval_path = args.out_path + args.exp_name + 'evaluation.txt'
            result_dict, result_ate_dict = evaluate(slam_path, gt_path)

            mono_relative_error_path = args.result_path + 'relative_error_' + str(i) + '.csv'
            result_df = pd.DataFrame.from_dict(result_dict, orient='index')

            mono_ate_error_path = args.result_path + 'ate_error_' + str(i) + '.csv'
            result_ate_df = pd.DataFrame.from_dict(result_ate_dict, orient='index')

            result_df.to_csv(mono_relative_error_path)
            result_ate_df.to_csv(mono_ate_error_path)

            mono_relative_list.append(result_df)
            mono_ate_list.append(result_ate_df)

        mono_relative_df = pd.concat(mono_relative_list, axis=1, ignore_index=True)
        mono_ate_df = pd.concat(mono_ate_list, axis=1, ignore_index=True)

        durations = defaultdict(list)
        for item in durations_list:
            for key, value in item.items():
                durations[key].append(value)
        durations_avg = {key: sum(vals) / len(vals) for key, vals in durations.items()}
        durations_avg_df = pd.DataFrame.from_dict(durations_avg, orient='index').T
        durations_df = pd.concat([durations_df, durations_avg_df], axis=0, ignore_index=True)

        # Mean #
        mono_relative_means_df = mono_relative_df.mean(axis=1).to_frame().T
        mono_ate_means_df = mono_ate_df.mean(axis=1).to_frame().T

        mono_relative_error_mean_path = args.out_path + args.optimization_base_type + '/relative_error.csv'
        mono_ate_error_mean_path = args.out_path + args.optimization_base_type + '/ate_error.csv'
        mono_relative_means_df.to_csv(mono_relative_error_mean_path)
        mono_ate_means_df.to_csv(mono_ate_error_mean_path)

        print(durations_avg_df)
        print(mono_ate_means_df)
        print(mono_relative_means_df)

        relative_df = pd.concat([mono_relative_means_df, relative_df], axis=0, ignore_index=True)
        ate_df = pd.concat([mono_ate_means_df, ate_df], axis=0, ignore_index=True)
        print('-----------------------------------------\n')

        # Virtual SLAM with gt depth #
        args.scale_aware = False
        args.network_depth = False
        args.optimization_base_type = 'virtual'
        mono_relative_list = []
        mono_ate_list = []
        durations_list = []

        print('[running virtual SLAM with gt depth]')
        for i in range(total_runs):
            print('\n')
            print(f'[iteration {i+1}]')

            args.exp_name = args.optimization_base_type + '_gt/' + args.optimization_base_type + '_gt_' + str(i) + '/'
            args.result_path = args.out_path + args.exp_name
            slam_path, durations_dict = main_loop(args)

            durations_list.append(durations_dict)

            eval_path = args.out_path + args.exp_name + 'evaluation.txt'
            result_dict, result_ate_dict = evaluate(slam_path, gt_path)

            mono_relative_error_path = args.result_path + 'relative_error_' + str(i) + '.csv'
            result_df = pd.DataFrame.from_dict(result_dict, orient='index')

            mono_ate_error_path = args.result_path + 'ate_error_' + str(i) + '.csv'
            result_ate_df = pd.DataFrame.from_dict(result_ate_dict, orient='index')

            result_df.to_csv(mono_relative_error_path)
            result_ate_df.to_csv(mono_ate_error_path)

            mono_relative_list.append(result_df)
            mono_ate_list.append(result_ate_df)

        mono_relative_df = pd.concat(mono_relative_list, axis=1, ignore_index=True)
        mono_ate_df = pd.concat(mono_ate_list, axis=1, ignore_index=True)

        durations = defaultdict(list)
        for item in durations_list:
            for key, value in item.items():
                durations[key].append(value)
        durations_avg = {key: sum(vals) / len(vals) for key, vals in durations.items()}
        durations_avg_df = pd.DataFrame.from_dict(durations_avg, orient='index').T
        durations_df = pd.concat([durations_df, durations_avg_df], axis=0, ignore_index=True)

        # Mean #
        mono_relative_means_df = mono_relative_df.mean(axis=1).to_frame().T
        mono_ate_means_df = mono_ate_df.mean(axis=1).to_frame().T

        mono_relative_error_mean_path = args.out_path + args.optimization_base_type + '_gt' + '/relative_error.csv'
        mono_ate_error_mean_path = args.out_path + args.optimization_base_type + '_gt' + '/ate_error.csv'
        mono_relative_means_df.to_csv(mono_relative_error_mean_path)
        mono_ate_means_df.to_csv(mono_ate_error_mean_path)

        print(durations_avg_df)
        print(mono_ate_means_df)
        print(mono_relative_means_df)

        relative_df = pd.concat([mono_relative_means_df, relative_df], axis=0, ignore_index=True)
        ate_df = pd.concat([mono_ate_means_df, ate_df], axis=0, ignore_index=True)
        print('-----------------------------------------\n')

        # Virtual SLAM #
        args.scale_aware = False
        args.network_depth = True
        args.optimization_base_type = 'virtual'
        mono_relative_list = []
        mono_ate_list = []
        durations_list = []

        print('[running virtual SLAM]')
        for i in range(total_runs):
            print('\n')
            print(f'[iteration {i+1}]')

            args.exp_name = args.optimization_base_type + '/' + args.optimization_base_type + '_' + str(i) + '/'
            args.result_path = args.out_path + args.exp_name
            slam_path, durations_dict = main_loop(args)

            durations_list.append(durations_dict)

            eval_path = args.out_path + args.exp_name + 'evaluation.txt'
            result_dict, result_ate_dict = evaluate(slam_path, gt_path)

            mono_relative_error_path = args.result_path + 'relative_error_' + str(i) + '.csv'
            result_df = pd.DataFrame.from_dict(result_dict, orient='index')

            mono_ate_error_path = args.result_path + 'ate_error_' + str(i) + '.csv'
            result_ate_df = pd.DataFrame.from_dict(result_ate_dict, orient='index')

            result_df.to_csv(mono_relative_error_path)
            result_ate_df.to_csv(mono_ate_error_path)

            mono_relative_list.append(result_df)
            mono_ate_list.append(result_ate_df)

        mono_relative_df = pd.concat(mono_relative_list, axis=1, ignore_index=True)
        mono_ate_df = pd.concat(mono_ate_list, axis=1, ignore_index=True)

        durations = defaultdict(list)
        for item in durations_list:
            for key, value in item.items():
                durations[key].append(value)
        durations_avg = {key: sum(vals) / len(vals) for key, vals in durations.items()}
        durations_avg_df = pd.DataFrame.from_dict(durations_avg, orient='index').T
        durations_df = pd.concat([durations_df, durations_avg_df], axis=0, ignore_index=True)

        # Mean #
        mono_relative_means_df = mono_relative_df.mean(axis=1).to_frame().T
        mono_ate_means_df = mono_ate_df.mean(axis=1).to_frame().T

        mono_relative_error_mean_path = args.out_path + args.optimization_base_type + '/relative_error.csv'
        mono_ate_error_mean_path = args.out_path + args.optimization_base_type + '/ate_error.csv'
        mono_relative_means_df.to_csv(mono_relative_error_mean_path)
        mono_ate_means_df.to_csv(mono_ate_error_mean_path)

        print(durations_avg_df)
        print(mono_ate_means_df)
        print(mono_relative_means_df)

        relative_df = pd.concat([mono_relative_means_df, relative_df], axis=0, ignore_index=True)
        ate_df = pd.concat([mono_ate_means_df, ate_df], axis=0, ignore_index=True)
        print('-----------------------------------------\n')

        # Global scale SLAM #
        args.scale_aware = True
        args.network_depth = True
        args.optimization_type = 'global'
        mono_relative_list = []
        mono_ate_list = []
        durations_list = []

        print('[running virtual SLAM]')
        for i in range(total_runs):
            print('\n')
            print(f'[iteration {i+1}]')

            args.exp_name = args.optimization_type + '/' + args.optimization_type + '_' + str(i) + '/'
            args.scale_path = args.optimization_type + '/' + args.optimization_type + '_' + str(i) + '/optimized_scale/'
            args.result_path = args.out_path + args.exp_name
            slam_path, durations_dict = main_loop(args)

            durations_list.append(durations_dict)

            eval_path = args.out_path + args.exp_name + 'evaluation.txt'
            result_dict, result_ate_dict = evaluate(slam_path, gt_path)

            mono_relative_error_path = args.result_path + 'relative_error_' + str(i) + '.csv'
            result_df = pd.DataFrame.from_dict(result_dict, orient='index')

            mono_ate_error_path = args.result_path + 'ate_error_' + str(i) + '.csv'
            result_ate_df = pd.DataFrame.from_dict(result_ate_dict, orient='index')

            result_df.to_csv(mono_relative_error_path)
            result_ate_df.to_csv(mono_ate_error_path)

            mono_relative_list.append(result_df)
            mono_ate_list.append(result_ate_df)

        mono_relative_df = pd.concat(mono_relative_list, axis=1, ignore_index=True)
        mono_ate_df = pd.concat(mono_ate_list, axis=1, ignore_index=True)

        durations = defaultdict(list)
        for item in durations_list:
            for key, value in item.items():
                durations[key].append(value)
        durations_avg = {key: sum(vals) / len(vals) for key, vals in durations.items()}
        durations_avg_df = pd.DataFrame.from_dict(durations_avg, orient='index').T
        durations_df = pd.concat([durations_df, durations_avg_df], axis=0, ignore_index=True)

        # Mean #
        mono_relative_means_df = mono_relative_df.mean(axis=1).to_frame().T
        mono_ate_means_df = mono_ate_df.mean(axis=1).to_frame().T

        mono_relative_error_mean_path = args.out_path + args.optimization_type + '/relative_error.csv'
        mono_ate_error_mean_path = args.out_path + args.optimization_type + '/ate_error.csv'
        mono_relative_means_df.to_csv(mono_relative_error_mean_path)
        mono_ate_means_df.to_csv(mono_ate_error_mean_path)

        print(durations_avg_df)
        print(mono_ate_means_df)
        print(mono_relative_means_df)

        relative_df = pd.concat([mono_relative_means_df, relative_df], axis=0, ignore_index=True)
        ate_df = pd.concat([mono_ate_means_df, ate_df], axis=0, ignore_index=True)
        print('-----------------------------------------\n')

        # Per-Class SLAM #
        args.scale_aware = True
        args.network_depth = True
        args.optimization_type = 'per_class'
        mono_relative_list = []
        mono_ate_list = []
        durations_list = []

        print('[running virtual SLAM]')
        for i in range(total_runs):
            print('\n')
            print(f'[iteration {i+1}]')

            args.exp_name = args.optimization_type + '/' + args.optimization_type + '_' + str(i) + '/'
            args.scale_path = args.optimization_type + '/' + args.optimization_type + '_' + str(i) + '/optimized_scale/'
            args.result_path = args.out_path + args.exp_name
            slam_path, durations_dict = main_loop(args)

            durations_list.append(durations_dict)

            eval_path = args.out_path + args.exp_name + 'evaluation.txt'
            result_dict, result_ate_dict = evaluate(slam_path, gt_path)

            mono_relative_error_path = args.result_path + 'relative_error_' + str(i) + '.csv'
            result_df = pd.DataFrame.from_dict(result_dict, orient='index')

            mono_ate_error_path = args.result_path + 'ate_error_' + str(i) + '.csv'
            result_ate_df = pd.DataFrame.from_dict(result_ate_dict, orient='index')

            result_df.to_csv(mono_relative_error_path)
            result_ate_df.to_csv(mono_ate_error_path)

            mono_relative_list.append(result_df)
            mono_ate_list.append(result_ate_df)

        mono_relative_df = pd.concat(mono_relative_list, axis=1, ignore_index=True)
        mono_ate_df = pd.concat(mono_ate_list, axis=1, ignore_index=True)

        durations = defaultdict(list)
        for item in durations_list:
            for key, value in item.items():
                durations[key].append(value)
        durations_avg = {key: sum(vals) / len(vals) for key, vals in durations.items()}
        durations_avg_df = pd.DataFrame.from_dict(durations_avg, orient='index').T
        durations_df = pd.concat([durations_df, durations_avg_df], axis=0, ignore_index=True)

        # Mean #
        mono_relative_means_df = mono_relative_df.mean(axis=1).to_frame().T
        mono_ate_means_df = mono_ate_df.mean(axis=1).to_frame().T

        mono_relative_error_mean_path = args.out_path + args.optimization_type + '/relative_error.csv'
        mono_ate_error_mean_path = args.out_path + args.optimization_type + '/ate_error.csv'
        mono_relative_means_df.to_csv(mono_relative_error_mean_path)
        mono_ate_means_df.to_csv(mono_ate_error_mean_path)

        print(durations_avg_df)
        print(mono_ate_means_df)
        print(mono_relative_means_df)

        relative_df = pd.concat([mono_relative_means_df, relative_df], axis=0, ignore_index=True)
        ate_df = pd.concat([mono_ate_means_df, ate_df], axis=0, ignore_index=True)
        print('-----------------------------------------\n')

        print('[final results]')

        relative_df.insert(0, 'method', methods_names[::-1])
        ate_df.insert(0, 'method', methods_names[::-1])
        durations_df.insert(0, 'method', methods_names)

        relative_df['scene'] = scene
        ate_df['scene'] = scene
        durations_df['scene'] = scene

        print('[relative pose errors]')
        print(relative_df)
        print('[absolute trajectory errors]')
        print(ate_df)
        print('[duration (sec)]')
        print(durations_df)
