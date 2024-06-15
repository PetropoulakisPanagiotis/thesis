import open3d as o3d
import numpy as np

def compute_accuracy(pcd_1, pcd_2):
    distances = pcd_2.compute_point_cloud_distance(pcd_1)
    return np.mean(distances), np.std(distances), np.max(distances), np.min(distances)

if __name__ == '__main__':

    pcd_1 = o3d.io.read_point_cloud("/home/petropoulakis/Desktop/thesis/code/thesis/rgbd_ptam/results/gt/map.pcd")
    min_bound = np.min(np.asarray(pcd_1.points), axis=0)
    max_bound = np.max(np.asarray(pcd_1.points), axis=0)
    voxel_size = 0.01
    grid_dim = ((max_bound - min_bound) / voxel_size).astype(int) + 1


    pcd_2 = o3d.io.read_point_cloud("/home/petropoulakis/Desktop/thesis/code/thesis/rgbd_ptam/results/exp_1/map.pcd")
    pcd_3 = o3d.io.read_point_cloud("/home/petropoulakis/Desktop/thesis/code/thesis/rgbd_ptam/results/gt_network/map.pcd")

    voxel_grid_1 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_1, voxel_size)
    voxel_grid_2 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_2, voxel_size)
    voxel_grid_3 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_3, voxel_size)

    print(compute_accuracy(pcd_2, pcd_1))
    print(compute_accuracy(pcd_3, pcd_1))
