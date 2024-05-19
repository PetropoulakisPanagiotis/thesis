from collections import namedtuple
import numpy as np
import g2o


class BundleAdjustment(g2o.SparseOptimizer):
    def __init__(self, ):
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

    def optimize(self, max_iterations=10):
        super().initialize_optimization()
        super().optimize(max_iterations)
        try:
            return not self.aborted
        finally:
            self.aborted = False

    def add_pose(self, pose_id: int, pose: g2o.SE3Quat, cam: namedtuple, fixed: bool = False) -> None:

        sbacam = g2o.CustomCam(pose.orientation(), pose.position())
        sbacam.set_cam(cam.fx, cam.fy, cam.cx, cam.cy)

        v_se3 = g2o.VertexCustomCam()
        v_se3.set_id(pose_id)
        v_se3.set_estimate(sbacam)
        v_se3.set_fixed(fixed)

        super().add_vertex(v_se3)

    def add_point(self, point_id: int, point: np.ndarray, fixed: bool = False, marginalized: bool = True) -> None:
        v_p = g2o.VertexCustomXYZ()
        v_p.set_id(point_id)
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

        edge.set_id(edge_id)
        edge.set_vertex(0, self.vertex(point_id))
        edge.set_vertex(1, self.vertex(pose_id))
        kernel = g2o.RobustKernelHuber(self.delta)
        edge.set_robust_kernel(kernel)
        super().add_edge(edge)

    def add_depth_scale_consistency_edge(self, edge_id: int, point_id: int, pose_id: int, scale_id: int, meas: float,
                                         information: np.ndarray = np.identity(1)) -> None:
        edge = g2o.EdgeDepthConsistencyScale()
        edge.set_id(edge_id)
        edge.set_measurement(meas)
        edge.set_information(information)

        # 0 Cam
        # 1 3D point
        # 2 scale
        edge.set_vertex(0, self.vertex(pose_id))
        edge.set_vertex(1, self.vertex(point_id))
        edge.set_vertex(2, self.vertex(scale_id))
        kernel = g2o.RobustKernelHuber(self.delta)
        edge.set_robust_kernel(kernel)
        super().add_edge(edge)

    def get_estimate(self, vertex_id: int) -> np.ndarray:
        return self.vertex(vertex_id).estimate()

    def abort(self):
        self.aborted = True


class LocalBA(object):
    def __init__(self, ):
        self.optimizer = BundleAdjustment()

        # threshold for confidence interval of 95%
        self.huber_threshold = 5.991
        self.total_points = 0

    def set_data(self, pose: g2o.SE3Quat, cam: namedtuple, points: np.ndarray, pixels: np.ndarray,
                 canonical_depth: np.ndarray, scale_network: float = 1, scale: float = 1) -> None:
        self.clear()

        self.optimizer.add_pose(0, pose, cam)
        self.optimizer.add_scale(1, scale)
        self.total_points = points.shape[0]
        for ii, point in enumerate(points):
            self.optimizer.add_point(ii + 2, point)
            u, v = pixels[:, ii]
            self.optimizer.add_camera_edge(ii, ii + 2, 0, [u, v])

        self.optimizer.add_scale(self.total_points + 3, scale)
        self.optimizer.add_scale_edge(self.total_points + 1, self.total_points + 3, scale_network)

        for ii, point in enumerate(points):
            self.optimizer.add_depth_scale_consistency_edge(self.total_points + 2, ii + 2, 0, self.total_points + 3,
                                                            canonical_depth[ii])

    def get_bad_measurements(self):
        bad_measurements = []
        for edge in self.optimizer.active_edges():
            print(edge.chi2())
            if edge.chi2() > self.huber_threshold:
                bad_measurements.append(edge.id())
        return bad_measurements

    def get_poses(self) -> list[np.ndarray]:
        return [self.optimizer.get_estimate(0)]

    def get_scales(self) -> list[np.ndarray]:
        return [self.optimizer.get_estimate(self.total_points + 3)]

    def clear(self):
        self.optimizer.clear()

    def abort(self):
        self.optimizer.abort()

    def optimize(self, max_iterations):
        return self.optimizer.optimize(max_iterations)
