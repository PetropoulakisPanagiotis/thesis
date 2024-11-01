import numpy as np
import cv2

from collections import defaultdict
from numbers import Number

from threading import Thread, Lock
from queue import Queue
"""
Image + features 
"""


class ImageFeature(object):
    def __init__(self, image, params):
        # TODO: pyramid representation
        self.image = image
        self.height, self.width = image.shape[:2]

        self.keypoints = []  # list of cv2.KeyPoint
        self.descriptors = []  # numpy.ndarray

        self.detector = params.feature_detector
        self.extractor = params.descriptor_extractor
        self.matcher = params.descriptor_matcher

        self.cell_size = params.matching_cell_size
        self.matching_distance = params.matching_distance
        self.neighborhood = (params.matching_cell_size * params.matching_neighborhood)

        self._lock = Lock()

    def extract(self, draw=False):
        self.keypoints = self.detector.detect(self.image)
        self.keypoints, self.descriptors = self.extractor.compute(self.image, self.keypoints)
        if draw:
            self.draw_keypoints()

        self.unmatched = np.ones(len(self.keypoints), dtype=bool)

    def draw_keypoints(self, name='keypoints', delay=0):
        if self.image.ndim == 2:
            image = np.repeat(self.image[..., np.newaxis], 3, axis=2)
        else:
            image = self.image
        img = cv2.drawKeypoints(image, self.keypoints, None, flags=0)
        cv2.imshow(name, img)
        cv2.waitKey(delay)

        #cv2.imwrite("./image_debug.png", img)

    def filter_features_to_class(self, mask, class_id, draw=False):   
        filtered_keypoints = []
        filtered_descriptors = []     
        for keypoint, descriptor in zip(self.keypoints, self.descriptors):
            xy = tuple(int(item) for item in keypoint.pt)
            if mask[xy[1], xy[0]] == class_id: 
                filtered_keypoints.append(keypoint)
                filtered_descriptors.append(descriptor)
    
        self.keypoints = filtered_keypoints
        self.descriptors = np.asarray(filtered_descriptors)
        self.unmatched = np.ones(len(self.keypoints), dtype=bool)

        if draw:
            self.draw_keypoints()    

    def find_matches(self, predictions, descriptors):
        matches = dict()
        distances = defaultdict(lambda: float('inf'))
        for m, query_idx, train_idx in self.matched_by(descriptors):
            if m.distance > min(distances[train_idx], self.matching_distance):
                continue

            pt1 = predictions[query_idx]
            pt2 = self.keypoints[train_idx].pt
            dx = pt1[0] - pt2[0]
            dy = pt1[1] - pt2[1]
            if np.sqrt(dx * dx + dy * dy) > self.neighborhood:
                continue

            matches[train_idx] = query_idx
            distances[train_idx] = m.distance
        matches = [(i, j) for j, i in matches.items()]
        return matches

    # Other ideas: row, col constraints #
    def matched_by(self, descriptors):
        with self._lock:
            unmatched_descriptors = self.descriptors[self.unmatched]
            if len(unmatched_descriptors) == 0:
                return []

            lookup = dict(zip(range(len(unmatched_descriptors)), np.where(self.unmatched)[0]))

        matches = self.matcher.match(np.array(descriptors), unmatched_descriptors)
        return [(m, m.queryIdx, m.trainIdx) for m in matches]

    def direct_match(self, *args, **kwargs):
        return direct_match(self.matcher, *args, **kwargs)

    def get_keypoint(self, i):
        return self.keypoints[i]

    def get_descriptor(self, i):
        return self.descriptors[i]

    def get_color(self, pt):
        x = int(np.clip(pt[0], 0, self.width - 1))
        y = int(np.clip(pt[1], 0, self.height - 1))
        color = self.image[y, x]
        return color[::-1] / 255.

    def set_matched(self, i):
        with self._lock:
            self.unmatched[i] = False

    def get_unmatched_keypoints(self):
        keypoints = []
        descriptors = []
        indices = []

        with self._lock:
            for i in np.where(self.unmatched)[0]:
                keypoints.append(self.keypoints[i])
                descriptors.append(self.descriptors[i])
                indices.append(i)

        return keypoints, descriptors, indices


def direct_match(matcher, desps1, desps2, matching_distance=30, ratio=0.7):
    matches = dict()
    distances = defaultdict(lambda: float('inf'))

    # Two matches for each desps1 #
    for (m, n) in matcher.knnMatch(np.array(desps1), np.array(desps2), k=2):
        if m.distance < min(matching_distance, n.distance * ratio, distances[m.trainIdx]):
            matches[m.trainIdx] = m.queryIdx
            distances[m.trainIdx] = m.distance

    return [(i, j) for j, i in matches.items()]
