import numpy as np
from utils.utils import invert_transformation_matrix
from scipy.spatial.transform import Rotation as R


def RTE(gt_pose: np.ndarray, pose: np.ndarray) -> float:
    return np.linalg.norm(gt_pose[:3, 3] - pose[:3, 3])

def RRE(gt_pose: np.ndarray, pose: np.ndarray) -> float:
    dt = invert_transformation_matrix(gt_pose) @ pose
    angle_axis = R.from_matrix(dt[:3, :3]).as_rotvec()
    rot = np.linalg.norm(angle_axis)
    return rot


def ATE_rot(gt_pose: np.ndarray, pose: np.ndarray) -> float:
    dt = gt_pose[:3, :3].T * pose[:3, :3]
    angle_axis = R.from_matrix(dt).as_rotvec()
    rot = np.linalg.norm(angle_axis)
    return rot

def ATE_trans(gt_pose: np.ndarray, pose: np.ndarray) -> float:
    return np.sum(np.abs((gt_pose[:3, 3] - pose[:3, 3])))
