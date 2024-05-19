import numpy as np
import cv2
import open3d as o3d
from collections import namedtuple
from scipy.spatial.transform import Rotation as R


def reprojection_error_test(u: float, v: float, cam: namedtuple,
        point_w: np.ndarray, w_transformation_c) -> float:
    point_w_h = np.append(point_w, 1)
    point_c_h = np.dot(w_transformation_c, point_w_h)

    error = [np.inf, np.inf]
    error[0] = u - ((cam.fx*point_c_h[0]/point_c_h[2]) + cam.cx)
    error[1] = v - ((cam.fy*point_c_h[1]/point_c_h[2]) + cam.cy)

    return error

def generate_colors(num_colors: int) -> np.ndarray:
    colors = np.random.rand(num_colors, 3)
    return colors


def invert_transformation_matrix(T: np.ndarray) -> np.ndarray:
    rotation = T[:3, :3]
    translation = T[:3, 3]
    inv_rotation = rotation.T
    inv_translation = -inv_rotation @ translation
    inv_T = np.eye(4)
    inv_T[:3, :3] = inv_rotation
    inv_T[:3, 3] = inv_translation
    return inv_T


def is_valid_rotation_matrix(R: np.ndarray) -> bool:
    should_be_identity = np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    should_be_one = np.isclose(np.linalg.det(R), 1.0, atol=1e-6)
    return should_be_identity and should_be_one


def visualize_matching_point_clouds(points_original: np.ndarray, points_perturb: np.ndarray) -> None:
    num_matches = points_original.shape[0]
    colors = generate_colors(num_matches)

    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(points_original)

    pcd_perturb = o3d.geometry.PointCloud()
    pcd_perturb.points = o3d.utility.Vector3dVector(points_perturb)
    # Update colors
    pcd_original.colors = o3d.utility.Vector3dVector(colors)
    pcd_perturb.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point clouds
    o3d.visualization.draw_geometries([pcd_original, pcd_perturb])


def visualize_depth_map(depth_map: np.ndarray) -> None:
    # Normalize depth values to the range [0, 255]
    min_depth = np.min(depth_map[depth_map > 0])  # Ignore zero (invalid depth)
    max_depth = np.max(depth_map)
    norm_depth = (depth_map - min_depth) / (max_depth - min_depth)
    norm_depth = (norm_depth * 255).astype(np.uint8)

    # Apply a colormap: closer points are darker, distant points are lighter
    colormap = cv2.COLORMAP_JET
    depth_colored = cv2.applyColorMap(norm_depth, colormap)

    # Display the depth map
    cv2.imshow("Depth Map Visualization", depth_colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_test_points_pixel_and_world_coords(cam: namedtuple, camera_to_world_transform: np.ndarray, image: np.ndarray,
                                           depth_map: np.ndarray, num_points: int = 50, offset: int = 50,
                                           min_depth: float = 0.1, max_depth: float = 10,
                                           viz: bool = False) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth_map.shape

    u = np.random.randint(0 + offset, width - offset, num_points)
    v = np.random.randint(0 + offset, height - offset, num_points)

    depth = depth_map[v, u]
    valid = (depth > min_depth) & (depth < max_depth)

    u = u[valid]
    v = v[valid]

    point_cloud_w = get_point_cloud_from_pixels(cam, u, v, depth_map, camera_to_world_transform)

    pixels = np.vstack((u, v))

    if viz:
        image_local = image.copy()
        for i in range(len(u)):
            cv2.circle(image_local, (u[i], v[i]), 3, (0, 255, 0), -1)
        cv2.imshow("Selected Pixels", image_local)
        cv2.waitKey(0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_w)
        o3d.visualization.draw_geometries([pcd])

    return pixels, point_cloud_w


def get_point_cloud_from_pixels(cam: namedtuple, u: np.ndarray, v: np.ndarray, depth: np.ndarray,
                                transformation_matrix: np.ndarray) -> np.ndarray:
    x = ((u - cam.cx) * depth[v, u]) / cam.fx
    y = ((v - cam.cy) * depth[v, u]) / cam.fy
    z = depth[v, u]

    point_cloud_c = np.vstack((x, y, z, np.ones_like(z)))
    point_cloud_t = transformation_matrix @ point_cloud_c

    return point_cloud_t[:3, :].reshape(-1, 3)


def normalize_angles(angles: np.ndarray) -> np.ndarray:
    # Ensure angles are within the range [-180, 180) degrees
    angles = (angles + 180) % 360 - 180
    return angles


def perturb_transformation_matrix(transformation_matrix: np.ndarray, translation_perturbation: np.ndarray,
                                  rotation_perturbation: np.ndarray) -> np.ndarray:
    rotation_matrix = transformation_matrix[:3, :3]
    translation_vector = transformation_matrix[:3, 3]

    original_rotation = R.from_matrix(rotation_matrix)

    original_euler = original_rotation.as_euler('xyz', degrees=True)

    perturbed_translation_vector = translation_vector + translation_perturbation
    perturbed_euler = normalize_angles(original_euler + rotation_perturbation)

    perturbed_rotation_matrix = R.from_euler('xyz', perturbed_euler, degrees=True).as_matrix()

    perturbed_transformation_matrix = np.eye(4)
    perturbed_transformation_matrix[:3, :3] = perturbed_rotation_matrix
    perturbed_transformation_matrix[:3, 3] = perturbed_translation_vector

    return perturbed_transformation_matrix
