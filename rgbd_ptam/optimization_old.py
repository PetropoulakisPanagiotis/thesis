from collections import namedtuple
import numpy as np
import g2o

from components import Camera

class BundleAdjustmentScaleAware(g2o.SparseOptimizer):
    def __init__(self, max_frames=50, max_instances=50):
        super().__init__()

        # Higher confident (better than CHOLMOD, according to
        # paper "3-D Mapping With an RGB-D Camera")
        solver = g2o.BlockSolverX(g2o.LinearSolverCSparseX())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

        # Convergence Criterion
        terminate = g2o.SparseOptimizerTerminateAction()
        terminate.set_gain_threshold(1e-6)
        super().add_post_iteration_action(terminate)

        # Robust cost Function (Huber function) delta
        self.delta = np.sqrt(5.991)
        self.aborted = False

        self.scale_offset = max_frames * max_instances

    def optimize(self, max_iterations=10):
        super().initialize_optimization()
        super().optimize(max_iterations)
        try:
            return not self.aborted
        finally:
            self.aborted = False

    def add_pose(self, pose_id: int, pose: g2o.SE3Quat, cam: Camera, fixed: bool = False) -> None:

        sbacam = g2o.CustomCam(pose.orientation(), pose.position())
        sbacam.set_cam(cam.fx, cam.fy, cam.cx, cam.cy)

        v_se3 = g2o.VertexCustomCam()
        v_se3.set_id(self.scale_offset + 2 * pose_id)
        v_se3.set_estimate(sbacam)
        v_se3.set_fixed(fixed)

        super().add_vertex(v_se3)

    def add_point(self, point_id: int, point: np.ndarray, fixed: bool = False, marginalized: bool = True) -> None:
        v_p = g2o.VertexCustomXYZ()
        v_p.set_id(self.scale_offset + 2 * point_id + 1)
        v_p.set_marginalized(marginalized)
        v_p.set_estimate(point)
        v_p.set_fixed(fixed)
        super().add_vertex(v_p)

    def add_scale(self, scale_id: int, scale: float, fixed: bool = False, marginalized: bool = False):
        scale_v = g2o.VertexScale()
        scale_v.set_id(scale_id)
        scale_v.set_estimate(scale)
        super().add_vertex(scale_v)

    def add_scale_edge(self, edge_id: int, scale_id: int, meas: float,
                       information: np.ndarray = np.identity(1)) -> None:
        edge = g2o.EdgeScaleNetworkConsistency()
        edge.set_measurement(meas)
        edge.set_information(information)

        edge.set_id(edge_id)
        edge.set_vertex(0, self.vertex(scale_id))
        kernel = g2o.RobustKernelHuber(self.delta)
        edge.set_robust_kernel(kernel)
        super().add_edge(edge)

    def add_camera_edge(self, edge_id: int, point_id: int, pose_id: int, meas: np.ndarray,
                        information: np.ndarray = np.identity(2)) -> None:
        edge = g2o.EdgeCustomCamera()
        edge.set_measurement(meas)
        edge.set_information(information)

        edge.set_id(self.scale_offset + 2 * edge_id)
        edge.set_vertex(0, self.vertex(self.scale_offset + 2*point_id + 1))
        edge.set_vertex(1, self.vertex(self.scale_offset + 2*pose_id))
        kernel = g2o.RobustKernelHuber(self.delta)
        edge.set_robust_kernel(kernel)
        super().add_edge(edge)

    def add_depth_scale_consistency_edge(self, edge_id: int, point_id: int, pose_id: int, scale_id: int, meas: float,
                                         information: np.ndarray = np.identity(1)) -> None:
        edge = g2o.EdgeDepthConsistencyScale()
        edge.set_id(self.scale_offset + 2 * edge_id + 1)
        edge.set_measurement(meas)
        edge.set_information(information)

        # 0 Cam
        # 1 3D point
        # 2 scale
        edge.set_vertex(0, self.vertex(self.scale_offset + 2*pose_id))
        edge.set_vertex(1, self.vertex(self.scale_offset + 2*point_id+1))
        edge.set_vertex(2, self.vertex(scale_id))
        kernel = g2o.RobustKernelHuber(self.delta)
        edge.set_robust_kernel(kernel)
        super().add_edge(edge)

    def get_pose(self, id):
        return self.vertex(self.scale_offset + id * 2).estimate()

    def get_point(self, id):
        return self.vertex(self.scale_offset + id * 2 + 1).estimate()

    def get_scale(self, id):
        return self.vertex(id).estimate()

    def get_scale_offset(self):
        return self.scale_offset

    def abort(self):
        self.aborted = True


class BundleAdjustment(g2o.SparseOptimizer):
    def __init__(self, ):
        super().__init__()

        # Higher confident (better than CHOLMOD, according to
        # paper "3-D Mapping With an RGB-D Camera")
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

        # Convergence Criterion
        terminate = g2o.SparseOptimizerTerminateAction()
        terminate.set_gain_threshold(1e-6)
        super().add_post_iteration_action(terminate)

        # Robust cost Function (Huber function) delta
        self.delta = np.sqrt(5.991)  # Power 2 is applied internally in g2o
        self.aborted = False

    def optimize(self, max_iterations=10):
        super().initialize_optimization()
        super().optimize(max_iterations)
        try:
            return not self.aborted
        finally:
            self.aborted = False

    # 0, 2, 4, 6, ...
    def add_pose(self, pose_id, pose, cam, fixed=False):
        sbacam = g2o.SBACam(pose.orientation(), pose.position())
        sbacam.set_cam(cam.fx, cam.fy, cam.cx, cam.cy, cam.baseline)

        v_se3 = g2o.VertexCam()
        v_se3.set_id(pose_id * 2)
        v_se3.set_estimate(sbacam)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

    # 1, 3, 5, 7
    def add_point(self, point_id, point, fixed=False, marginalized=True):
        v_p = g2o.VertexSBAPointXYZ()
        v_p.set_id(point_id * 2 + 1)
        v_p.set_marginalized(marginalized)
        v_p.set_estimate(point)
        v_p.set_fixed(fixed)
        super().add_vertex(v_p)

    def add_edge(self, id, point_id, pose_id, meas):
        if meas.is_stereo():
            edge = self.stereo_edge(meas.xyx)
        elif meas.is_left():
            edge = self.mono_edge(meas.xy)
        edge.set_id(id)
        edge.set_vertex(0, self.vertex(point_id * 2 + 1))
        edge.set_vertex(1, self.vertex(pose_id * 2))
        kernel = g2o.RobustKernelHuber(self.delta)
        edge.set_robust_kernel(kernel)
        super().add_edge(edge)

    def stereo_edge(self, projection, information=np.identity(3)):
        e = g2o.EdgeProjectP2SC()
        e.set_measurement(projection)
        e.set_information(information)
        return e

    def mono_edge(self, projection, information=np.identity(2)):
        e = g2o.EdgeProjectP2MC()
        e.set_measurement(projection)
        e.set_information(information)
        return e

    def get_pose(self, id):
        return self.vertex(id * 2).estimate()

    def get_point(self, id):
        return self.vertex(id * 2 + 1).estimate()

    def abort(self):
        self.aborted = True


"""
For local map
"""


class LocalBA(object):
    def __init__(self, ):
        self.optimizer = BundleAdjustment()
        self.measurements = []  # Measurements, can be both from optimized non-optimzed pose frames
        self.keyframes = []  # Frames that their pose is optimized
        self.mappoints = set()  # Only add 3D points that belong to frames that their pose is optimized

        # threshold for confidence interval of 95%
        self.huber_threshold = 5.991

    def set_data(self, adjust_keyframes, fixed_keyframes):
        self.clear()  # Empty buffers
        # Note: for edge_id, we add the id of the self.measurements list #

        for kf in adjust_keyframes:
            self.optimizer.add_pose(kf.id, kf.pose, kf.cam, fixed=False)
            self.keyframes.append(kf)

            for m in kf.measurements():
                pt = m.mappoint

                # Add 3D point: this point can be visible in many keyframes                  #
                # But this 3D point can not be matched with many 2D points on the same frame #
                if pt not in self.mappoints:
                    self.optimizer.add_point(pt.id, pt.position)
                    self.mappoints.add(pt)

                # Add pose - 2D measurement constraint alogn with 3D point #
                edge_id = len(self.measurements)
                self.optimizer.add_edge(edge_id, pt.id, kf.id, m)
                self.measurements.append(m)

        for kf in fixed_keyframes:
            self.optimizer.add_pose(kf.id, kf.pose, kf.cam, fixed=True)

            # Local 3D points can be visible to frames outside the window #
            # Add these pose contraints                                   #
            for m in kf.measurements():
                if m.mappoint in self.mappoints:
                    edge_id = len(self.measurements)
                    self.optimizer.add_edge(edge_id, m.mappoint.id, kf.id, m)
                    self.measurements.append(m)

    def update_points(self):
        for mappoint in self.mappoints:
            mappoint.update_position(self.optimizer.get_point(mappoint.id))

    def update_poses(self):
        for keyframe in self.keyframes:
            keyframe.update_pose(self.optimizer.get_pose(keyframe.id))
            # Update pose constraints                #
            # From current to reference or preceding #
            keyframe.update_reference()
            keyframe.update_preceding()

    def get_bad_measurements(self):
        bad_measurements = []
        for edge in self.optimizer.active_edges():
            if edge.chi2() > self.huber_threshold:
                bad_measurements.append(self.measurements[edge.id()])
        return bad_measurements

    def clear(self):
        self.optimizer.clear()
        self.keyframes.clear()
        self.mappoints.clear()
        self.measurements.clear()

    def abort(self):
        self.optimizer.abort()

    def optimize(self, max_iterations):
        return self.optimizer.optimize(max_iterations)


class LocalBAScaleAware(object):
    def __init__(self, ):
        self.optimizer = BundleAdjustmentScaleAware()
        self.measurements = {}  # Measurements, can be both from optimized non-optimzed pose frames
        self.keyframes = []     # Frames that their pose is optimized
        self.mappoints = set()  # Only add 3D points that belong to frames that their pose is optimized
        self.scales_start = [0]  # Each frame has a scale, save indexes for easy mapping 

        # threshold for confidence interval of 95%
        self.huber_threshold = 5.991

    def set_data(self, adjust_keyframes, fixed_keyframes):
        self.clear()  # Empty buffers

        # Note: for edge_id, we add the id of the self.measurements list #
        for kf in adjust_keyframes:
            # Pose id: scale_offset + 2*id
            self.optimizer.add_pose(kf.id, kf.pose, kf.cam, fixed=False)
            self.keyframes.append(kf)
            # Add scales #
            for ii, scale in enumerate(kf.scale_aware_frame.scales):
                # Scale Id: ii + scale_start
                self.optimizer.add_scale(self.scales_start[-1] + ii, scale, fixed=False)

                # Edge Id: ii + scale_start
                self.optimizer.add_scale_edge(self.scales_start[-1] + ii, self.scales_start[-1] + ii, scale,
                                              information=np.identity(1) * 1./kf.scale_aware_frame.scales_uncertainty[ii])

            for m in kf.measurements():
                pt = m.mappoint

                # Add 3D point: this point can be visible in many keyframes                  #
                # But this 3D point can not be matched with many 2D points on the same frame #
                if pt not in self.mappoints:
                    self.optimizer.add_point(pt.id, pt.position)
                    self.mappoints.add(pt)

                # Add pose - 2D measurement constraint alogn with 3D point #
                edge_id = len(self.measurements)
                self.optimizer.add_camera_edge(edge_id, pt.id, kf.id, m.xy)
                self.measurements[2*edge_id + self.optimizer.scale_offset] = m

                # Edge Id: scale_offset + 2*ii + 1
                self.optimizer.add_depth_scale_consistency_edge(ii, ii, kf.id, self.scales_start[-1] + m.scale_id_measurement, m.canonical_measurement,
                                                                information=np.identity(1) * 1./m.covariance_canonical_measurement)

            self.scales_start.append(self.scales_start[-1] + len(kf.scale_aware_frame.scales))

        for kf in fixed_keyframes:
            self.optimizer.add_pose(kf.id, kf.pose, kf.cam, fixed=True)

            # Add scales #
            for ii, scale in enumerate(kf.scale_aware_frame.scales):
                # Scale Id: ii + scale_start
                self.optimizer.add_scale(self.scales_start[-1] + ii, scale, fixed=True)
                # Edge Id: ii + scale_start
                self.optimizer.add_scale_edge(self.scales_start[-1] + ii, self.scales_start[-1] + ii, scale,
                                              information=np.identity(1) * 1./kf.scale_aware_frame.scales_uncertainty[ii])


            # Local 3D points can be visible to frames outside the window #
            # Add these pose contraints                                   #
            for m in kf.measurements():
                if m.mappoint in self.mappoints:
                    edge_id = len(self.measurements)
                    self.optimizer.add_camera_edge(edge_id, m.mappoint.id, kf.id, m.xy)
                    self.measurements[2*edge_id + self.optimizer.scale_offset] = m

                    # Edge Id: scale_offset + 2*ii + 1
                    self.optimizer.add_depth_scale_consistency_edge(ii, ii, kf.id, self.scales_start[-1] + m.scale_id_measurement, m.canonical_measurement,
                                                                    information=np.identity(1) * 1./m.covariance_canonical_measurement)

            self.scales_start.append(self.scales_start[-1] + len(kf.scale_aware_frame.scales))

    def update_points(self):
        for mappoint in self.mappoints:
            mappoint.update_position(self.optimizer.get_point(mappoint.id))

    def update_poses(self):
        for keyframe in self.keyframes:
            keyframe.update_pose(self.optimizer.get_pose(keyframe.id))
            # Update pose constraints                #
            # From current to reference or preceding #
            keyframe.update_reference()
            keyframe.update_preceding()

    def update_scales(self):
        for ii, keyframe in enumerate(self.keyframes):
            scales = []
            for jj in range(len(keyframe.scale_aware_frame.scales)):
                scale = self.optimizer.get_scale(self.scales_start[ii] + jj)
                scales.append(scale)
            keyframe.update_scale(scales)

    def get_bad_measurements(self):
        bad_measurements = []
        for edge in self.optimizer.active_edges():
            if isinstance(edge, g2o.EdgeCustomCamera):
                if edge.chi2() > self.huber_threshold:
                    bad_measurements.append(self.measurements[edge.id()])
        return bad_measurements

    def clear(self):
        self.optimizer.clear()
        self.keyframes.clear()
        self.mappoints.clear()
        self.measurements.clear()
        self.scales_start = [0]

    def abort(self):
        self.optimizer.abort()

    def optimize(self, max_iterations):
        return self.optimizer.optimize(max_iterations)


class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

    def optimize(self, max_iterations=20):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_vertex(self, id, pose, fixed=False):
        v_se3 = g2o.VertexSE3()
        v_se3.set_id(id)
        v_se3.set_estimate(pose)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

    def add_edge(self, vertices, measurement=None, information=np.identity(6), robust_kernel=None):

        edge = g2o.EdgeSE3()

        # Two vertices #
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)

        if measurement is None:
            measurement = (edge.vertex(0).estimate().inverse() * edge.vertex(1).estimate())

        edge.set_measurement(measurement)
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)

        super().add_edge(edge)

    def set_data(self, keyframes, loops):
        super().clear()

        # Find earliest frame in loop #
        anchor = None
        for kf, *_ in loops:
            if anchor is None or kf < anchor:
                anchor = kf

        # Keyframes should be corrected since we change the pose in the loop #
        for i, kf in enumerate(keyframes):
            pose = g2o.Isometry3d(kf.orientation, kf.position)

            # Fix pose of first frame             #
            # Fix pose of frames less than anchor #
            fixed = i == 0
            if anchor is not None:
                fixed = kf <= anchor

            self.add_vertex(kf.id, pose, fixed=fixed)

            if kf.preceding_keyframe is not None:
                self.add_edge(vertices=(kf.preceding_keyframe.id, kf.id), measurement=kf.preceding_constraint)

            if (kf.reference_keyframe is not None and kf.reference_keyframe != kf.preceding_keyframe):
                self.add_edge(vertices=(kf.reference_keyframe.id, kf.id), measurement=kf.reference_constraint)

        # Add loop frames #
        for kf, kf2, meas in loops:
            self.add_edge((kf.id, kf2.id), measurement=meas)

    def update_poses_and_points(self, keyframes, correction=None, exclude=set()):
        # Correct keyframes with new pose estimates #
        for kf in keyframes:
            if len(exclude) > 0 and kf in exclude:
                continue

            uncorrected = g2o.Isometry3d(kf.orientation, kf.position)
            if correction is None:  # Use optimization result
                vertex = self.vertex(kf.id)
                if vertex.fixed():
                    continue

                corrected = vertex.estimate()
            else:
                corrected = uncorrected * correction

            # Correction is minor discard #
            delta = uncorrected.inverse() * corrected
            if (g2o.AngleAxis(delta.rotation()).angle() < 0.02
                    and np.linalg.norm(delta.translation()) < 0.03):  # 1°, 3cm
                continue

            for m in kf.measurements():
                if m.from_triangulation():
                    old = m.mappoint.position
                    # TWC * TCW * W_P
                    new = corrected * (uncorrected.inverse() * old)
                    m.mappoint.update_position(new)
                    # TODO: update normal?

            kf.update_pose(corrected)
