import numpy as np
import cv2
import g2o

from threading import Lock, Thread
from queue import Queue

from enum import Enum
from collections import defaultdict

from covisibility import GraphKeyFrame
from covisibility import GraphMapPoint
from covisibility import GraphMeasurement
"""
Camera intrinsics
"""


class Camera(object):
    def __init__(self, fx, fy, cx, cy, width, height, scale, baseline, depth_near, depth_far, frustum_near,
                 frustum_far):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale
        self.baseline = baseline
        self.bf = fx * baseline

        self.intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        self.depth_near = depth_near  # 0.1
        self.depth_far = depth_far  # 10
        self.frustum_near = frustum_near  # 0.1
        self.frustum_far = frustum_far  # 50

        self.width = width
        self.height = height


"""
ScaleAware: canonical, scales + uncertainties
"""


class ScaleAwareFrame(object):
    def __init__(self, idx, canonical=None, canonical_uncertainty=None, scales=None, scales_uncertainty=None,
                 pixel_to_scale_map=None, scales_valid=None):
        self.idx = idx
        self.canonical = canonical  # 'image'
        if canonical is not None:
            self.height, self.width = canonical.shape[:2]
        else:
            self.height, self.width = 0, 0

        self.canonical_uncertainty = canonical_uncertainty  # 'image'
        self.scales = scales  # list
        self.scales_uncertainty = scales_uncertainty  # list
        self.pixel_to_scale_map = pixel_to_scale_map  # 'image'
        self.scales_valid = scales_valid  # valid scale list 0 or 1


"""
Frame:
       image, pose, cam, timestamp, projection
       feature: image, keypoints, desc
"""


class Frame(object):
    def __init__(self, idx, pose, feature, cam, timestamp=None, pose_covariance=np.identity(6)):
        self.idx = idx
        self.pose = pose  # g2o.Isometry3d
        self.feature = feature
        self.cam = cam
        self.timestamp = timestamp
        self.image = feature.image

        self.orientation = pose.orientation()  # From camera to world
        self.position = pose.position()  # From camera to world
        self.pose_covariance = pose_covariance

        self.transform_matrix = pose.inverse().matrix()[:3]  # Shape: (3, 4)
        self.projection_matrix = (self.cam.intrinsic.dot(self.transform_matrix))  # From world frame to image

    def can_view(self, points, margin=10):
        # Frustum Culling (batch version)
        points = np.transpose(points)
        (u, v), depth = self.project(self.transform(points))
        if depth.max() > 20:
            exit()
        return np.logical_and.reduce([
            depth >= self.cam.frustum_near, depth <= self.cam.frustum_far, u >= -margin, u <= self.cam.width + margin, v
            >= -margin, v <= self.cam.height + margin
        ])

    def update_pose(self, pose):
        if isinstance(pose, g2o.SE3Quat):
            self.pose = g2o.Isometry3d(pose.orientation(), pose.position())
        else:
            self.pose = pose
        self.orientation = self.pose.orientation()
        self.position = self.pose.position()

        self.transform_matrix = self.pose.inverse().matrix()[:3]
        self.projection_matrix = (self.cam.intrinsic.dot(self.transform_matrix))

    def transform(self, points):
        '''
        Transform points from world coordinates frame to camera frame.
        Args:
            points: a point or an array of points, of shape (3,) or (3, N).
        '''
        assert len(points) > 0
        R = self.transform_matrix[:3, :3]
        if points.ndim == 1:
            t = self.transform_matrix[:3, 3]
        else:
            t = self.transform_matrix[:3, 3:]
        return R.dot(points) + t

    def project(self, points):
        '''
        Project points from camera frame to image's pixel coordinates.
        Args:
            points: a point or an array of points, of shape (3,) or (3, N).
        Returns:
            Projected pixel coordinates, and respective depth.
        '''
        projection = self.cam.intrinsic.dot(points / points[-1:])
        return projection[:2], points[-1]

    def find_matches(self, points, descriptors):
        '''
        Match to points from world frame.
        Args:
            points: a list/array of points. shape: (N, 3)
            descriptors: a list of feature descriptors. length: N
        Returns:
            List of successfully matched (queryIdx, trainIdx) pairs.
        '''
        points = np.transpose(points)
        proj, _ = self.project(self.transform(points))
        proj = proj.transpose()
        return self.feature.find_matches(proj, descriptors)

    def get_keypoint(self, i):
        return self.feature.get_keypoint(i)

    def get_descriptor(self, i):
        return self.feature.get_descriptor(i)

    def get_color(self, pt):
        return self.feature.get_color(pt)

    def set_matched(self, i):
        self.feature.set_matched(i)

    def get_unmatched_keypoints(self):
        return self.feature.get_unmatched_keypoints()


def depth_to_3d(depth, coords, cam):
    coords = np.array(coords, dtype=int)
    ix = coords[:, 0]
    iy = coords[:, 1]
    depth = depth[iy, ix]

    zs = depth / cam.scale
    xs = (ix - cam.cx) * zs / cam.fx
    ys = (iy - cam.cy) * zs / cam.fy
    return np.column_stack([xs, ys, zs])


"""
Frame +  depth
rgb is FRAME
"""


class RGBDFrame(Frame, ScaleAwareFrame):
    def __init__(self, idx, pose, feature, depth, cam, timestamp=None, pose_covariance=np.identity(6), canonical=None,
                 canonical_uncertainty=None, scales=None, scales_uncertainty=None, pixel_to_scale_map=None,
                 scales_valid=None):
        super().__init__(idx, pose, feature, cam, timestamp, pose_covariance)
        self.rgb = Frame(idx, pose, feature, cam, timestamp, pose_covariance)
        self.scale_aware_frame = None
        if canonical is not None:
            self.scale_aware_frame = ScaleAwareFrame(idx, canonical / cam.scale, canonical_uncertainty, scales,
                                                     scales_uncertainty, pixel_to_scale_map, scales_valid)
        self.depth = depth
    def virtual_stereo(self, px):
        x, y = int(px[0]), int(px[1])
        if not (0 <= x <= self.cam.width - 1 and 0 <= y <= self.cam.height - 1):
            return None

        depth = self.depth[y, x] / self.cam.scale
        if not (self.cam.depth_near <= depth <= self.cam.depth_far):
            return None

        disparity = self.cam.bf / depth

        # virtual right camera observation
        kp2 = cv2.KeyPoint(x - disparity, y, 1)

        return kp2

    def find_matches(self, source, points, descriptors):
        # Find matches and get measurements
        # Source: type like TRACKING - enum
        matches = self.rgb.find_matches(points, descriptors)
        measurements = []
        for i, j in matches:
            px = self.rgb.get_keypoint(j).pt
            kp2 = self.virtual_stereo(px)

            xy = tuple(int(item) for item in self.rgb.get_keypoint(j).pt)
            canonical_measurement = None
            covariance_canonical_measurement = None
            scale_id_measurement = None
            if self.scale_aware_frame is not None:
                canonical_measurement = self.scale_aware_frame.canonical[xy[1], xy[0]]
                covariance_canonical_measurement = self.scale_aware_frame.canonical_uncertainty[xy[1], xy[0]]
                scale_id_measurement = self.scale_aware_frame.pixel_to_scale_map[xy[1], xy[0]]

            if kp2 is not None:
                measurement = Measurement(Measurement.Type.STEREO, source, [self.rgb.get_keypoint(j), kp2],
                                          [self.rgb.get_descriptor(j)] * 2, canonical_measurement,
                                          covariance_canonical_measurement, scale_id_measurement)
            else:  # Typically when depth is 0 for this camera
                measurement = Measurement(Measurement.Type.LEFT, source, [self.rgb.get_keypoint(j)],
                                          [self.rgb.get_descriptor(j)], canonical_measurement,
                                          covariance_canonical_measurement, scale_id_measurement)
            measurements.append((i, measurement))
            self.rgb.set_matched(j)
        return measurements

    def match_mappoints(self, mappoints, source):
        points = []
        descriptors = []
        for mappoint in mappoints:
            points.append(mappoint.position)
            descriptors.append(mappoint.descriptor)
        matched_measurements = self.find_matches(source, points, descriptors)

        measurements = []
        # For a current measurement we attach a mappoint             #
        # We will update later the descriptor once the current frame #
        # can see this mappoint point                                #
        for i, meas in matched_measurements:
            meas.mappoint = mappoints[i]
            measurements.append(meas)
        return measurements

    def match_mappoints_and_get_ids(self, mappoints, source):
        points = []
        descriptors = []
        for mappoint in mappoints:
            points.append(mappoint.position)
            descriptors.append(mappoint.descriptor)
        matched_measurements = self.find_matches(source, points, descriptors)

        measurements = []
        ids = []
        # For a current measurement we attach a mappoint             #
        # We will update later the descriptor once the current frame #
        # can see this mappoint point                                #
        for i, meas in matched_measurements:
            meas.mappoint = mappoints[i]
            measurements.append(meas)
            ids.append(i)
        return measurements, ids

    def cloudify(self):
        # Create new mappoints if not matched #
        # Return mappoints and measurements   #
        kps, desps, idx = self.rgb.get_unmatched_keypoints()
        px = np.array([kp.pt for kp in kps])
        if len(px) == 0:
            return [], []

        pts = depth_to_3d(self.depth, px, self.cam)
        Rt = self.pose.matrix()[:3]
        R = Rt[:, :3]
        t = Rt[:, 3:]
        points = (R.dot(pts.T) + t).T  # world frame

        mappoints = []
        measurements = []
        for i, point in enumerate(points):
            # Invalid depth #
            if not (self.cam.depth_near <= pts[i][2] <= self.cam.depth_far):
                continue

            kp2 = self.virtual_stereo(px[i])
            # Invalid depth for virtual stereo   #
            # Only mappoints with virtual stereo #
            if kp2 is None:
                continue

            normal = point - self.position
            normal /= np.linalg.norm(normal)
            color = self.rgb.get_color(px[i])

            xy = tuple(int(item) for item in kps[i].pt)
            canonical_measurement = None
            covariance_canonical_measurement = None
            scale_id_measurement = None

            if self.scale_aware_frame is not None:
                canonical_measurement = self.scale_aware_frame.canonical[xy[1], xy[0]]
                covariance_canonical_measurement = self.scale_aware_frame.canonical_uncertainty[xy[1], xy[0]]
                scale_id_measurement = self.scale_aware_frame.pixel_to_scale_map[xy[1], xy[0]]

            # Triangulation aka virtual stereo #
            mappoint = MapPoint(point, normal, desps[i], color)
            measurement = Measurement(Measurement.Type.STEREO, Measurement.Source.TRIANGULATION, [kps[i], kp2],
                                      [desps[i], desps[i]], canonical_measurement, covariance_canonical_measurement,
                                      scale_id_measurement)

            measurement.mappoint = mappoint
            mappoints.append(mappoint)
            measurements.append(measurement)

            self.rgb.set_matched(i)

        return mappoints, measurements

    def update_pose(self, pose):
        super().update_pose(pose)
        self.rgb.update_pose(pose)

    def update_scale(self, scales: list):
        if self.scale_aware_frame is not None:
            for ii, (scale, scale_valid) in enumerate(zip(scales, self.scale_aware_frame.scales_valid)):
                if scale_valid == 1:
                    self.scale_aware_frame.scales[ii] = scale

    def can_view(self, mappoints):  # batch version
        points = []
        point_normals = []
        for i, p in enumerate(mappoints):
            points.append(p.position)
            point_normals.append(p.normal)

        points = np.asarray(points)
        point_normals = np.asarray(point_normals)

        normals = points - self.position
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

        # cos(theta) = dot(a,b) / norm(a) * norm(b)                     #
        # TODO: 'Compare normal of mappoint with current normal camera' #
        cos = np.clip(np.sum(point_normals * normals, axis=1), -1, 1)

        parallel = np.arccos(cos) < (np.pi / 4)

        # Depth based #
        can_view = self.rgb.can_view(points)

        return np.logical_and(parallel, can_view)

    def to_keyframe(self):
        if self.scale_aware_frame is not None:
            keyframe = KeyFrame(self.idx, self.pose, self.feature, self.depth, self.cam, self.timestamp,
                            self.pose_covariance)
            keyframe.scale_aware_frame = self.scale_aware_frame
            return keyframe
        else:
            return KeyFrame(self.idx, self.pose, self.feature, self.depth, self.cam, self.timestamp,
                            self.pose_covariance)


"""
GraphKeyFrame
Frame +  depth
rgb is FRAME
"""


class KeyFrame(GraphKeyFrame, RGBDFrame):
    _id = 0
    _id_lock = Lock()

    def __init__(self, *args, **kwargs):
        GraphKeyFrame.__init__(self)
        RGBDFrame.__init__(self, *args, **kwargs)
        with KeyFrame._id_lock:
            self.id = KeyFrame._id
            KeyFrame._id += 1

        self.reference_keyframe = None  # Frame with most matches
        self.reference_constraint = None  # From the reference to this frame
        self.preceding_keyframe = None  # Previous keyframe
        self.preceding_constraint = None
        self.loop_keyframe = None
        self.loop_constraint = None
        self.fixed = False

    def update_reference(self, reference=None):
        if reference is not None:
            self.reference_keyframe = reference
        self.reference_constraint = (self.reference_keyframe.pose.inverse() * self.pose)

    def update_preceding(self, preceding=None):
        if preceding is not None:
            self.preceding_keyframe = preceding
        self.preceding_constraint = (self.preceding_keyframe.pose.inverse() * self.pose)

    def set_loop(self, keyframe, constraint):
        self.loop_keyframe = keyframe
        self.loop_constraint = constraint

    def is_fixed(self):
        return self.fixed

    def set_fixed(self, fixed=True):
        self.fixed = fixed


class MapPoint(GraphMapPoint):
    _id = 0
    _id_lock = Lock()

    def __init__(self, position, normal, descriptor, color=np.zeros(3), covariance=np.identity(3) * 1e-4):
        super().__init__()

        with MapPoint._id_lock:
            self.id = MapPoint._id
            MapPoint._id += 1

        self.position = position
        self.normal = normal
        self.descriptor = descriptor
        self.covariance = covariance
        self.color = color

        self.count = defaultdict(int)  # TODO: see puprose this

    def update_position(self, position):
        self.position = position

    def update_normal(self, normal):
        self.normal = normal

    def update_descriptor(self, descriptor):
        self.descriptor = descriptor

    def set_color(self, color):
        self.color = color

    # TODO: check this #
    def is_bad(self):
        with self._lock:
            status = (self.count['meas'] == 0
                      or (self.count['outlier'] > 20 and self.count['outlier'] > self.count['inlier'])
                      or (self.count['proj'] > 50 and self.count['proj'] > self.count['meas'] * 50))
            return status

    def increase_outlier_count(self):
        with self._lock:
            self.count['outlier'] += 1

    # TODO: inliner always 0 #
    def increase_inlier_count(self):
        with self._lock:
            self.count['inlier'] += 1

    def increase_projection_count(self):
        with self._lock:
            self.count['proj'] += 1

    def increase_measurement_count(self):
        with self._lock:
            self.count['meas'] += 1


class Measurement(GraphMeasurement):

    Source = Enum('Measurement.Source', ['TRIANGULATION', 'TRACKING', 'REFIND'])
    Type = Enum('Measurement.Type', ['STEREO', 'LEFT', 'RIGHT'])

    def __init__(self, type, source, keypoints, descriptors, canonical_measurement=1,
                 covariance_canonical_measurement=np.identity(1), scale_id_measurement=0):
        super().__init__()

        self.type = type
        self.source = source
        self.keypoints = keypoints  # List: can be left, right
        self.descriptors = descriptors
        self.xy = np.array(self.keypoints[0].pt)

        if self.is_stereo():
            self.xyx = np.array([*keypoints[0].pt, keypoints[1].pt[0]])

        self.triangulation = (source == self.Source.TRIANGULATION)

        # Canonical measurement #
        self.canonical_measurement = canonical_measurement
        self.covariance_canonical_measurement = covariance_canonical_measurement
        self.scale_id_measurement = scale_id_measurement

    def get_canonical(self):
        return self.canonical_measurement

    def get_covariance_canonical_measurement(self):
        return self.covariance_canonical_measurement

    def get_scale_id_measurement(self):
        return self.scale_id_measurement

    def get_descriptor(self, i=0):
        return self.descriptors[i]

    def get_keypoint(self, i=0):
        return self.keypoints[i]

    def get_descriptors(self):
        return self.descriptors

    def get_keypoints(self):
        return self.keypoints

    def is_stereo(self):
        return self.type == Measurement.Type.STEREO

    def is_left(self):
        return self.type == Measurement.Type.LEFT

    def is_right(self):
        return self.type == Measurement.Type.RIGHT

    def from_triangulation(self):
        return self.triangulation

    def from_tracking(self):
        return self.source == Measurement.Source.TRACKING

    def from_refind(self):
        return self.source == Measurement.Source.REFIND
