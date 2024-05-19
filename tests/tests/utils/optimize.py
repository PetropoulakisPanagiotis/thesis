from collections import namedtuple
import numpy as np
import g2o


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
        v_p = g2o.VertexSBAPointXYZ()#g2o.VertexCustomXYZ()
        #v_p = g2o.VertexCustomXYZ()
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

    def add_camera_edge(self, edge_id: int, point_id: int, pose_id: int, meas: np.ndarray,
                        information: np.ndarray = np.identity(2)) -> None:
        #edge = g2o.EdgeStereo()
        edge = g2o.EdgeProjectP2MC()
        edge.set_measurement(meas)
        edge.set_information(information)

        edge.set_id(edge_id)
        edge.set_vertex(0, self.vertex(point_id))
        edge.set_vertex(1, self.vertex(pose_id))
        kernel = g2o.RobustKernelHuber(self.delta)
        edge.set_robust_kernel(kernel)
        super().add_edge(edge)

    def add_edge_depth_consistency(self, pose, cam, point):
        id = 5000000
        sbacam = g2o.CustomCam(pose.orientation(), pose.position())
        sbacam.set_cam(cam.fx, cam.fy, cam.cx, cam.cy, cam.baseline)

        v_se3 = g2o.VertexCustomCam()
        v_se3.set_id(id)
        v_se3.set_estimate(sbacam)
        v_se3.set_fixed(False)
        super().add_vertex(v_se3)

        v_p = g2o.VertexCustomXYZ()

        v_p.set_id(id + 1)
        v_p.set_marginalized(True)
        v_p.set_estimate(point)
        v_p.set_fixed(False)
        super().add_vertex(v_p)

        c = g2o.VertexCanonical()
        c.set_id(id + 2)
        c.set_estimate(5)
        c.set_fixed(False)
        super().add_vertex(c)

        s = g2o.VertexScale()
        s.set_id(id + 3)
        s.set_estimate(5)
        s.set_fixed(False)
        super().add_vertex(s)

        e = g2o.EdgeDepthConsistencyScale()
        e.set_id(id + 4)
        e.set_measurement(5)
        e.set_information(np.identity(1))

        # 0 cam: Pose + t
        # 1 3D point: XYZ
        # 2 canonical: C
        # 3 scale: S
        e.set_vertex(0, self.vertex(id))
        e.set_vertex(1, self.vertex(id + 2))
        e.set_vertex(2, self.vertex(id + 3))
        kernel = g2o.RobustKernelHuber(self.delta)
        e.set_robust_kernel(kernel)

    def get_estimate(self, vertex_id: int) -> np.ndarray:
        return self.vertex(vertex_id).estimate()

    def abort(self):
        self.aborted = True


class LocalBA(object):
    def __init__(self, ):
        self.optimizer = BundleAdjustment()

        # threshold for confidence interval of 95%
        self.huber_threshold = 5.991

    def set_data(self, pose: g2o.SE3Quat, cam: namedtuple, points: np.ndarray, pixels: np.ndarray,
                 canonical_depth: np.ndarray, scale: float = 1) -> None:
        self.clear()

        self.optimizer.add_pose(0, pose, cam)
        self.optimizer.add_scale(1, scale)

        for ii, point in enumerate(points):
            self.optimizer.add_point(ii + 2, point)
            u, v = pixels[:, ii]
            self.optimizer.add_camera_edge(ii, ii + 2, 0, [u, v])

    def get_bad_measurements(self):
        bad_measurements = []
        for edge in self.optimizer.active_edges():
            print(edge.chi2())
            if edge.chi2() > self.huber_threshold:
                bad_measurements.append(edge.id())
        return bad_measurements

    def get_poses(self) -> list[np.ndarray]:
        return [self.optimizer.get_estimate(0)]

    def clear(self):
        self.optimizer.clear()

    def abort(self):
        self.optimizer.abort()

    def optimize(self, max_iterations):
        return self.optimizer.optimize(max_iterations)
