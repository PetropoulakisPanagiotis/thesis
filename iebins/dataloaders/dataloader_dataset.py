import os
import json
import random

import copy

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset

from utils import _is_pil_image, _is_numpy_image


class DatasetPreprocess(Dataset):
    def __init__(self, args, mode, transform=None):
        self.args = args

        if self.args.dataset == 'scannet':
            if mode == 'train':
                with open(args.filenames_file, 'r') as f:
                    self.filenames = f.read().splitlines()
            else:
                with open(args.filenames_file_eval, 'r') as f:
                    self.filenames = f.read().splitlines()
        else: # nyu
            if mode == 'train':
                with open(args.filenames_file, 'r') as f:
                    self.filenames = f.readlines()
            else:
                with open(args.filenames_file_eval, 'r') as f:
                    self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensorCustom

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        if self.mode == 'train':
            # Open images #
            if self.args.dataset == 'scannet':
                rgb_path = self.args.data_path + "rgb/" + sample_path + ".jpg"
                rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
                image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                depth_path = self.args.data_path + "depth/" + sample_path + ".png"
                depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            else:
                rgb_file = sample_path.split()[0]
                depth_file = sample_path.split()[1]

                image_path = os.path.join(self.args.data_path, rgb_file)
                depth_path = os.path.join(self.args.gt_path, depth_file)
                image = Image.open(image_path)
                depth_gt = Image.open(depth_path)

            # NYU masks and instances #
            if self.args.dataset == 'nyu' and self.args.segmentation:
                annotations_file = os.path.join(self.args.data_path, sample_path.split()[0])
                annotations_file = annotations_file.replace("rgb", "annotations")
                annotations_file = annotations_file[:len(annotations_file) - 4] + ".json"

                instances_masks, segmentation_map, instances_labels, instances_bbox, instances_areas, num_semantic_classes = \
                                 load_image_annotations_nyu(annotations_file)

            # To avoid blank boundaries due to pixel registration #
            if self.args.dataset == 'nyu' and self.args.dataset != 'scannet':
                depth_gt = np.array(depth_gt)
                valid_mask = np.zeros_like(depth_gt)
                valid_mask[45:472, 43:608] = 1
                depth_gt[valid_mask == 0] = 0
                depth_gt = Image.fromarray(depth_gt)

            # Normalize image #
            image = np.asarray(image, dtype=np.float32) / 255.0

            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)

            # De-normalize depth #
            depth_gt = depth_gt / 1000.0

            # General augmentations #
            image, depth_gt = self.train_preprocess(image, depth_gt, self.args)

            sample = {'image': image, 'depth': depth_gt, 'has_valid_depth': True}
        else:
            if self.args.dataset == 'scannet':
                rgb_path = self.args.data_path_eval + "rgb/" + sample_path + ".jpg"
                rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
                image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                image = np.asarray(image, dtype=np.float32) / 255.0

                depth_path = self.args.data_path_eval + "depth/" + sample_path + ".png"
                depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                has_valid_depth = True
            else: # NYU
                data_path = self.args.data_path_eval
                image_path = os.path.join(data_path, sample_path.split()[0])

                # Normalize #
                image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

                if 'eval' in self.mode:
                    gt_path = self.args.gt_path_eval
                    depth_path = os.path.join(gt_path, sample_path.split()[1])

                    has_valid_depth = False
                    try:
                        depth_gt = Image.open(depth_path)
                        has_valid_depth = True
                    except IOError:
                        depth_gt = False
                        print('Missing gt for {}'.format(image_path))
                        exit()

            if has_valid_depth:
                depth_gt = np.asarray(depth_gt, dtype=np.float32)
                depth_gt = np.expand_dims(depth_gt, axis=2)

                # De-normalize #
                depth_gt = depth_gt / 1000.0

            # NYU masks and instances #
            if self.args.dataset == 'nyu' and self.args.segmentation:
                annotations_file = os.path.join(data_path, sample_path.split()[0])
                annotations_file = annotations_file.replace("rgb", "annotations")
                annotations_file = annotations_file[:len(annotations_file) - 4] + ".json"

                instances_masks, segmentation_map, instances_labels, instances_bbox, instances_areas, \
                                 num_semantic_classes = load_image_annotations_nyu(annotations_file)

            if 'eval' in self.mode:
                sample = {'image': image, 'depth': depth_gt, 'has_valid_depth': has_valid_depth}
            else:
                sample = {'image': image}

        # Fill sample NYU #
        if self.args.dataset == 'nyu' and self.args.segmentation:
            sample['instances_masks'] = instances_masks
            sample['segmentation_map'] = segmentation_map
            sample['instances_labels'] = instances_labels
            sample['instances_bbox'] = instances_bbox
            sample['instances_areas'] = instances_areas

        if self.args.dataset == 'scannet':
            if self.mode == 'train':
                sample['dataset_path'] = self.args.data_path
            else:
                sample['dataset_path'] = self.args.data_path_eval

            sample['filename'] = sample_path

        # ToTensorCustom, also load scannet if dataset == scannet #
        if self.transform:
            sample = self.transform([sample, self.args.dataset])
        
        return sample

    # Random gamma, brightness, and color augmentations #
    def train_preprocess(self, image, depth_gt, args, segmentation_map=None):
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # Gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image**gamma

        # Brightness augmentation
        brightness = random.uniform(0.75, 1.25)

        image_aug = image_aug * brightness

        # Color augmentation - fixed
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image

        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)


class ToTensorCustom(object):
    def __init__(self, mode, segmentation, classes_mapping=None):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.max_instances = 65
        self.segmentation = segmentation
        self.classes_mapping = classes_mapping  # mapp original segmentation map to a smaller subset of classes

    def __call__(self, sample_dataset):

        sample = sample_dataset[0]
        dataset = sample_dataset[1]

        image = sample['image']
        depth = sample['depth']
        #image_numpy = image.copy()
       
        # Image #
        image = self.to_tensor(image)
        image = self.normalize(image)
        
        # Depth #
        depth = self.to_tensor(depth)

        result_dict = {}
        result_dict['image'] = image
        result_dict['depth'] = depth
        result_dict['has_valid_depth'] = sample['has_valid_depth']

        if dataset == 'nyu' and self.segmentation:
            offset = 0
            segmentation_map = torch.from_numpy(sample['segmentation_map'])

            instances_masks = torch.stack(
                [torch.from_numpy(arr.astype(np.float32)) for arr in sample['instances_masks']])

            # For null class create an instance #
            null_mask = (torch.sum(instances_masks, dim=0) == 0).to(dtype=torch.float32).unsqueeze(0)
            instances_masks = torch.cat((null_mask, instances_masks), dim=0)

            ys, xs = torch.where(null_mask.squeeze(0))
            xmin = torch.clamp(torch.min(xs) - offset, min=0)
            ymin = torch.clamp(torch.min(ys) - offset, min=0)
            xmax = torch.clamp(torch.max(xs) + offset, max=segmentation_map.shape[2] - 1)
            ymax = torch.clamp(torch.max(ys) + offset, max=segmentation_map.shape[1] - 1)

            # Padd invalid instances to create even batch #
            num_zeros_needed = self.max_instances - instances_masks.shape[0]
            zero_tensors = [
                torch.zeros(1, *instances_masks.shape[1:], dtype=torch.int32) for _ in range(num_zeros_needed)
            ]
            instances_masks = torch.cat([instances_masks] + zero_tensors, dim=0)

            # Labels #
            instances_labels = torch.from_numpy(np.insert(sample['instances_labels'], 0, [0], axis=0))
            minus_tensors = [-1 * torch.ones(1, dtype=torch.int32) for _ in range(num_zeros_needed)]
            instances_labels = torch.cat([instances_labels] + minus_tensors, dim=0)

            # Bboxes #
            sample['instances_bbox'] = np.insert(sample['instances_bbox'], 0, [xmin, ymin, xmax, ymax], axis=0)
            instances_bbox = torch.stack([torch.from_numpy(arr) for arr in sample['instances_bbox']])
            zero_tensors = [torch.zeros((1, 4), dtype=torch.int32) for _ in range(num_zeros_needed)]
            instances_bbox = torch.cat([instances_bbox] + zero_tensors, dim=0)

            # Areas #
            sample['instances_areas'] = np.insert(sample['instances_areas'], 0, [(xmax - xmin) * (ymax - ymin)], axis=0)
            instances_areas = torch.stack([torch.from_numpy(np.asarray([arr])) for arr in sample['instances_areas']])
            zero_tensors = [torch.zeros((1, 1), dtype=torch.int32) for _ in range(num_zeros_needed)]
            instances_areas = torch.cat([instances_areas] + zero_tensors, dim=0)
            
            debug = False
            if debug: 
                img = instances_masks[0].cpu().numpy().astype(np.uint8)
                img[img == 1] = [255]
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                bbox = sample['instances_bbox'][0]
                x1, y1, x2, y2 = bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow("instance", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            result_dict['instances_masks'] = instances_masks
            result_dict['segmentation_map'] = segmentation_map
            result_dict['instances_labels'] = instances_labels
            result_dict['instances_bbox'] = instances_bbox
            result_dict['instances_areas'] = instances_areas

        elif dataset == 'scannet' and self.segmentation:
            seg_map, instances_map, boxes, instances_labels = load_image_annotations_scannet(
                sample['dataset_path'], sample['filename'], mapping_40_to_13=self.classes_mapping)

            segmentation_map = torch.from_numpy(seg_map)
            instances_masks = torch.stack([torch.from_numpy(arr).to(torch.float32) for arr in instances_map])
            instances_bbox = torch.stack([torch.from_numpy(arr).to(torch.int32) for arr in boxes])
            instances_labels = torch.from_numpy(instances_labels).to(torch.long)

            result_dict['instances_masks'] = instances_masks
            result_dict['segmentation_map'] = segmentation_map
            result_dict['instances_labels'] = instances_labels
            result_dict['instances_bbox'] = instances_bbox

        return result_dict

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # Handle PIL Image #
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK #
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)

        img = img.view(pic.size[1], pic.size[0], nchannel)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()

        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


def load_image_annotations_nyu(json_file_path):
    with open(json_file_path, 'r') as file:
        coco_data = json.load(file)

    num_semantic_classes = int(coco_data.get("num_categories")) + 1  # null-class ~ 14

    bounding_boxes = []
    areas_instances = []
    instances = []  # 0/1
    labels_instances = []

    segmentation_map = np.array(coco_data.get("categories"), dtype=np.int32)
    segmentation_map = create_one_hot_mask_classes_np(segmentation_map, num_semantic_classes)
    _, h, w = segmentation_map.shape  # [num_classes, h, w] - 0/1

    # Iterate over annotations per image #
    for ii, annotation in enumerate(coco_data.get("annotations", [])):

        # Unpack instance mask #
        instance_pixels = np.array(annotation.get("segmentation", []), dtype=np.uint32)
        instance_map = np.zeros((h, w), dtype=np.int32)
        instance_map[instance_pixels[:, 1], instance_pixels[:, 0]] = 1
        instances.append(instance_map)

        # Category id #
        label = annotation.get("category_id", 0)
        labels_instances.append(label)

        # Unpack bbox #
        bbox = annotation.get("bbox", [])
        bounding_boxes.append(bbox)

        # Unpack area #
        area = annotation.get("area", [])
        areas_instances.append(area)

    return instances, segmentation_map, np.array(labels_instances), np.asarray(bounding_boxes, dtype=np.int32), \
           np.asarray(areas_instances, dtype=np.int32), num_semantic_classes


def load_image_annotations_scannet(dataset_path, filename, mapping_40_to_13, num_classes=14):
    seg_map, instances_map, boxes = None, None, None
    
    # Segmentation #
    seg_path = dataset_path + "semantic_refined_40/" + filename + ".png"
    seg_map_original = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)

    # + null -> 14 in total#
    seg_map_original = mapping_40_to_13[seg_map_original]  

    # (class, h, w) #
    seg_map = create_one_hot_mask_classes_np(seg_map_original, num_classes)

    # Instances #
    inst_path = dataset_path + "instance_refined/" + filename + ".png"
    instances_map = cv2.imread(inst_path, cv2.IMREAD_UNCHANGED).astype('int32')

    instances_map, boxes, instances_labels = create_instance_masks_and_boxes_scannet_np(
        seg_map_original, instances_map)

    return seg_map, instances_map, boxes, instances_labels


def create_one_hot_mask_classes_np(segmentation_map, num_classes):
    # segmentation_map: (h, w)
    # one_got_mask:     (classes, h, w)
    one_hot_mask = np.zeros((num_classes, *segmentation_map.shape), dtype=np.float32)
    
    debug = False
    if debug:
        data = custom_cmap_instances(segmentation_map / np.max(segmentation_map))  
        data = data[:, :] * 255
        cv2.imshow("colored classes", data[:, :].astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Set the corresponding class index to 1 #
    for class_idx in range(num_classes):
        one_hot_mask[class_idx] = (segmentation_map == class_idx).astype(np.float32)

    return one_hot_mask


def create_instance_masks_and_boxes_scannet_np(seg_map, instance_map, max_instances=65):
    unique_ids = np.unique(instance_map)
    masks = []
    boxes = []
    offset = 0
    labels = []

    # Create also an instance with null class @
    null_class_mask = np.uint8(instance_map == 0)

    # Iterate over the unique instance IDs #
    for instance_id in unique_ids:
        if instance_id == 0:  # Skip background
            continue

        instance_mask = np.uint8(instance_map == instance_id)
        instance_class = seg_map[np.where(instance_mask)][0]

        # Do not include void instances #
        if instance_class == 0:
            null_class_mask = np.logical_or(null_class_mask, instance_mask)
            continue

        masks.append(instance_mask)
        labels.append(instance_class)

        # Box #
        ys, xs = np.where(instance_mask)
        xmin = max(np.min(xs) - offset, 0)
        xmax = min(np.max(xs) + offset, instance_mask.shape[1] - 1)
        ymin = max(np.min(ys) - offset, 0)
        ymax = min(np.max(ys) + offset, instance_mask.shape[0] - 1)
        boxes.append(np.asarray([xmin, ymin, xmax, ymax]))
        
        debug = False
        if debug:    
            instance_mask[instance_mask == 1] = [255]
            instance_mask = cv2.cvtColor(instance_mask, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(instance_mask, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.imshow("Instance", instance_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Insert null class instance at the beginning #
    if not np.all(null_class_mask == 0):
        masks.insert(0, null_class_mask)
        labels.insert(0, 0)

        ys, xs = np.where(null_class_mask)
        xmin = max(np.min(xs) - offset, 0)
        xmax = min(np.max(xs) + offset, null_class_mask.shape[1] - 1)
        ymin = max(np.min(ys) - offset, 0)
        ymax = min(np.max(ys) + offset, null_class_mask.shape[0] - 1)
        boxes.insert(0, np.asarray([xmin, ymin, xmax, ymax]))
    
        debug = False
        if debug:
            null_class_mask[null_class_mask == True] = [255]
            null_class_mask = cv2.cvtColor(null_class_mask, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(null_class_mask, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.imshow("Null", null_class_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Make batch even #
    num_remaining = max_instances - len(masks)
    for _ in range(num_remaining):
        masks.append(np.zeros_like(instance_map, dtype=np.uint8))
        boxes.append(np.asarray([0, 0, 0, 0]))
        labels.append(-1)

    return masks, boxes, np.array(labels)
