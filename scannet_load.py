import numpy as np
import json
import cv2
from dataclasses import dataclass
from typing import Tuple, Union
from scipy.spatial.transform import Rotation
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import open3d as o3d 
import matplotlib.pyplot as plt

SCANNET_COLOR_MAP_20 = {
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    14: (247., 182., 210.),
    15: (66., 188., 102.),
    16: (219., 219., 141.),
    17: (140., 57., 197.),
    18: (202., 185., 52.),
    19: (51., 176., 203.),
    20: (200., 54., 131.),
    21: (92., 193., 61.),
    22: (78., 71., 183.),
    23: (172., 114., 82.),
    24: (255., 127., 14.),
    25: (91., 163., 138.),
    26: (153., 98., 156.),
    27: (140., 153., 101.),
    28: (158., 218., 229.),
    29: (100., 125., 154.),
    30: (178., 127., 135.),
    32: (146., 111., 194.),
    33: (44., 160., 44.),
    34: (112., 128., 144.),
    35: (96., 207., 209.),
    36: (227., 119., 194.),
    37: (213., 92., 176.),
    38: (94., 106., 211.),
    39: (82., 84., 163.),
    40: (100., 85., 144.),
}

# set missing colors in dictionary, for source of colors see:
# https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/util.py
SCANNET_COLOR_MAP_20[13] = (178, 76, 76)
SCANNET_COLOR_MAP_20[31] = (120, 185, 128)


CLASS_LABELS_20 = ('void', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                   'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                   'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture')

VALID_CLASS_IDS_20 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)

SEMANTIC_CLASS_COLORS_SCANNET_20 = tuple(
            tuple(int(val) for val in SCANNET_COLOR_MAP_20[idx])  # convert float to int
            for idx in ((0,) + (VALID_CLASS_IDS_20))
        )

class ScanNet():
    def __init__(self, dataset_path: str, split: str = 'train') -> None:

        self.dataset_path = dataset_path
        self.split = split
        self.semantic_n_classes = 20
        self.max_instances = 30

        file_names_path = self.dataset_path + "/" + split + ".txt"
        with open(file_names_path, 'r') as f:
            self.filenames = f.read().splitlines()

        self.intr = np.eye(3)
        self.intr[0][0] = 577.8706114969136
        self.intr[0][2] = 319.87654320987656
        self.intr[1][1] = 577.8706114969136
        self.intr[1][2] = 238.88888888888889

        self.color_mappings = {}

        for i in range(len(SEMANTIC_CLASS_COLORS_SCANNET_20)):
            self.color_mappings[i] = SEMANTIC_CLASS_COLORS_SCANNET_20[i]

        self.labels = CLASS_LABELS_20

    def viz(self, image, depth, seg, inst):
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        depth_normalized = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        depth_colormap = cv2.applyColorMap(np.uint8(255 * depth_normalized), cv2.COLORMAP_JET)

        cv2.imshow('Depth Colormap', depth_colormap)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Create an empty RGB image to hold the colored segmentation map
        segmentation_colored = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)

        # Apply the color mapping to the segmentation map
        for class_idx, color in self.color_mappings.items():
            segmentation_colored[seg == class_idx] = color
            print(class_idx)

        # Display the colored segmentation map
        cv2.imshow('Segmentation Map Colored', segmentation_colored)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        # Get unique instance IDs
        unique_ids = np.unique(inst)

        # Create an empty RGB image to hold the colored instance map
        instance_colored = np.zeros((inst.shape[0], inst.shape[1], 3), dtype=np.uint8)

        # Assign unique colors to each instance
        for instance_id in unique_ids:
            if instance_id == 0:  # Skip background
                continue

            # Generate a random color for each instance
            color = np.random.randint(0, 256, size=3)
 
            # Assign the color to pixels corresponding to the current instance
            instance_colored[inst == instance_id] = color

        # Display the colored instance map
        cv2.imshow('Instance Map Colored', instance_colored)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def get_item(self, idx):
        assert idx < len(self.filenames) and idx >= 0

        # rgb #
        rgb_path = self.dataset_path + "/" + self.split + "/rgb/" + self.filenames[idx] + ".jpg"
        rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)

        # depth #
        depth_path = self.dataset_path + "/" + self.split + "/depth/" + self.filenames[idx] + ".png"
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        #depth /= 1000

        # extr #
        extr_path = self.dataset_path + "/" + self.split + "/extrinsics/" + self.filenames[idx] + ".json"

        with open(extr_path, 'r') as f:
            extr = json.load(f)

        w, x, y, z = extr['quat_w'], extr['quat_x'], extr['quat_y'], extr['quat_z']
        # Assuming you have translation values (tx, ty, tz)
        tx = extr['x']
        ty = extr['y']
        tz = extr['z']
        # Create rotation matrix from quaternions
        rotation_matrix = Rotation.from_quat([x, y, z, w]).as_matrix()

        # Create translation matrix
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = [tx, ty, tz] 
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = [tx, ty, tz]
        extr = transformation_matrix

        sem, inst, sem_map, inst_map, boxes = None, None, None, None, None
        if self.split != 'test':
            # sem #
            sem_path = self.dataset_path + "/" + self.split + "/semantic_refined_" + str(self.semantic_n_classes) + "/" + self.filenames[idx] + ".png"
            sem = cv2.imread(sem_path, cv2.IMREAD_UNCHANGED)

            # inst #
            inst_path = self.dataset_path + "/" + self.split + "/instance_refined/" + self.filenames[idx] + ".png"
            inst = cv2.imread(inst_path, cv2.IMREAD_UNCHANGED).astype('int32')

            sem_map = create_one_hot_mask_classes_np(sem, self.semantic_n_classes + 1)
            inst_map, boxes = create_instance_masks_and_boxes(sem, inst, self.max_instances)

        return rgb, depth, extr, sem, inst, sem_map, inst_map, boxes

def create_one_hot_mask_classes_np(segmentation_map, num_classes):
    # segmentation_map: (h, w)
    # (classes, h, w)
    one_hot_mask = np.zeros((num_classes, *segmentation_map.shape), dtype=np.float32)

    # Set the corresponding class index to 1
    for class_idx in range(num_classes):
        one_hot_mask[class_idx] = (segmentation_map == class_idx).astype(np.float32)
 
    return one_hot_mask

def create_instance_masks_and_boxes(seg_map, instance_map, max_instances=32):
    # Get unique instance IDs including 0 (background)
    unique_ids = np.unique(instance_map)
    # Initialize lists to hold binary masks and bounding boxes for each instance
    masks = []
    boxes = []
    offset = 2
    # Iterate over the unique instance IDs
    for instance_id in unique_ids:
        if instance_id == 0:  # Skip background
            continue


        instance_mask = np.uint8(instance_map == instance_id)
        # Do not include void instances 
        if seg_map[np.where(instance_mask)][0] == 0:
            continue

        # Find the coordinates of all pixels where instance ID is present
        ys, xs = np.where(instance_mask)
        # Calculate bounding box coordinates
        xmin = max(np.min(xs) - offset, 0)
        xmax = min(np.max(xs) + offset, instance_mask.shape[1] - 1)
        ymin = max(np.min(ys) - offset, 0)
        ymax = min(np.max(ys) + offset, instance_mask.shape[0] - 1)
        masks.append(instance_mask)
        boxes.append((xmin, ymin, xmax, ymax))


        """
        instance_mask[instance_mask == 1] = [255]
        instance_mask = cv2.cvtColor(instance_mask, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(instance_mask, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.imshow("instance", instance_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

    # Append zero masks and boxes for remaining instances if needed
    num_remaining = max_instances - len(masks)
    for _ in range(num_remaining):
        masks.append(np.zeros_like(instance_map, dtype=np.uint8))
        boxes.append((0, 0, 0, 0))

    return masks, boxes

if __name__ == '__main__':

    dataset = ScanNet('/home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted')
    rgb_1, depth_1, extr_1, sem, inst, sem_map, inst_map, boxes = dataset.get_item(0)
    rgb_2, depth_2, extr_2, sem_1, inst_1, sem_map_1, inst_map_1, boxes_1 = dataset.get_item(20)
    #dataset.viz(rgb, depth, sem, inst)

    image_1_name = "00000"
    scene_name = "scene0191_00"

    print(rgb_1.shape)
    #depth_1 = o3d.io.read_image("/home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted/train/depth/" + scene_name + "/" + image_1_name + ".png")
    #rgb_1 = o3d.io.read_image("/home/petropoulakis/Desktop/thesis/code/datasets/scannet/data_converted/train/rgb/" + scene_name + "/" + image_1_name + ".jpg")
    rgb_1 = o3d.geometry.Image(rgb_1)
    depth_1 = o3d.geometry.Image(depth_1)

    rgb_2 = o3d.geometry.Image(np.array(rgb_2))
    depth_2 = o3d.geometry.Image(np.array(depth_2))

    intr = dataset.intr
    cam1 = o3d.camera.PinholeCameraIntrinsic()
    cam1.set_intrinsics(640, 480, intr[0][0], intr[1][1], intr[0][2], intr[1][2])
    print(cam1.intrinsic_matrix)
    rgbd_image_1 = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_1, depth_1, convert_rgb_to_intensity=False)
    pcd_1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_1, cam1, np.linalg.inv(extr_1))

    rgbd_image_2 = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_2, depth_2, convert_rgb_to_intensity=False)
    pcd_2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_2, cam1, np.linalg.inv(extr_2))
    pcd_2.colors = o3d.utility.Vector3dVector(np.zeros_like(np.asarray(pcd_2.points)))

    # Flip it, otherwise the pointcloud will be upside down
    pcd_1.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd_2.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd_1, pcd_2])

