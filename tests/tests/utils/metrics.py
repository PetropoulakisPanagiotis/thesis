import numpy as np
from utils.utils import invert_transformation_matrix


def ATE(gt_pose: np.ndarray, pose: np.ndarray) -> tuple[float, np.ndarray]:
    PE = invert_transformation_matrix(pose) @ gt_pose

    return np.linalg.norm(gt_pose[:3, 3] - pose[:3, 3]) * 100, PE


def ARE(PE: np.ndarray) -> float:
    return np.arccos(np.trace(PE[:3, :3]) - 1 / 2) * 180 / np.pi * 100
