import os
import json
import random

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset

from utils_clean import _is_pil_image, _is_numpy_image, custom_cmap_instances


class DatasetPreprocess(Dataset):
    def __init__(self, args, mode, transform=None):
        self.args = args
        if mode == 'online_eval':
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensorCustom

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        # focal = float(sample_path.split()[2])
        focal = 518.8579

        if self.mode == 'train':
            if self.args.dataset == 'kitti':
                rgb_file = sample_path.split()[0]
                depth_file = os.path.join(sample_path.split()[1])

                if self.args.use_right is True and random.random() > 0.5:
                    rgb_file.replace('image_02', 'image_03')
                    depth_file.replace('image_02', 'image_03')
            else:
                rgb_file = sample_path.split()[0]
                depth_file = sample_path.split()[1]

            # Open images 
            image_path = os.path.join(self.args.data_path, rgb_file)
            depth_path = os.path.join(self.args.gt_path, depth_file)
            image = Image.open(image_path)
            depth_gt = Image.open(depth_path)

            if self.args.dataset == 'nyu' and self.args.segmentation:
                
                annotations_file = os.path.join(self.args.data_path, sample_path.split()[0])
                annotations_file = annotations_file.replace("rgb", "annotations")
                annotations_file = annotations_file[:len(annotations_file)-4] + ".json"

                instances_masks, segmentation_map, instances_labels, instances_bbox, instances_areas, num_semantic_classes = \
                                 load_image_annotations_nyu(annotations_file)

            # Not used in NYU
            if self.args.do_kb_crop is True: 
                height = image.height
                width = image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))


            # To avoid blank boundaries due to pixel registration
            if self.args.dataset == 'nyu' or self.args.dataset == 'nyud':
                if self.args.input_height == 480:
                    depth_gt = np.array(depth_gt)
                    valid_mask = np.zeros_like(depth_gt)

                    valid_mask[45:472, 43:608] = 1
                    depth_gt[valid_mask == 0] = 0

                    depth_gt = Image.fromarray(depth_gt)
                else:
                    print("Warn - height for nyu != 480")
                    depth_gt = depth_gt.crop((43, 45, 608, 472))
                    image = image.crop((43, 45, 608, 472))

            # Random rotate: not used in NYU
            if self.args.do_random_rotate is True and self.args.dataset != 'nyu':
                random_angle = (random.random() - 0.5) * 2 * self.args.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)

            # Normalize image
            image = np.asarray(image, dtype=np.float32) / 255.0

            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)

            # Fix depth range
            if self.args.dataset == 'nyu' or self.args.dataset == 'nyud':
                depth_gt = depth_gt / 1000.0
            else:
                depth_gt = depth_gt / 256.0

            # Random crop: not used in NYU
            if(image.shape[0] != self.args.input_height or image.shape[1] != self.args.input_width) and self.args.dataset != 'nyu':
                image, depth_gt = self.random_crop(image, depth_gt, self.args.input_height, self.args.input_width)

            # General augmentations
            if self.args.dataset != 'nyu' or self.args.segmentation == False:
                image, depth_gt = self.train_preprocess(image, depth_gt, self.args)
            else:
                image, depth_gt = self.train_preprocess(image, depth_gt, self.args, segmentation_map)

            # https://github.com/ShuweiShao/URCDC-Depth
            if self.args.dataset != 'nyu':
                image, depth_gt = self.Cut_Flip(image, depth_gt)

            sample = {'image': image, 'depth': depth_gt, 'focal': focal}
        else:
            if self.mode == 'online_eval':
                data_path = self.args.data_path_eval
            else:
                data_path = self.args.data_path

            image_path = os.path.join(data_path,  sample_path.split()[0])

            # Normalize
            image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

            if self.mode == 'online_eval':
                gt_path = self.args.gt_path_eval
                depth_path = os.path.join(gt_path, sample_path.split()[1])

                if self.args.dataset == 'kitti':
                    depth_path = os.path.join(gt_path, sample_path.split()[1])
                has_valid_depth = False
                try:
                    depth_gt = Image.open(depth_path)
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    #print('Missing gt for {}'.format(image_path))

                if has_valid_depth:
                    depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    depth_gt = np.expand_dims(depth_gt, axis=2)

                    # Fix depth ranges 
                    if self.args.dataset == 'nyu' or self.args.dataset == 'nyud':
                        depth_gt = depth_gt / 1000.0
                    else:
                        depth_gt = depth_gt / 256.0

            if self.args.dataset == 'nyu' and self.args.segmentation:
                annotations_file = os.path.join(data_path, sample_path.split()[0])
                annotations_file = annotations_file.replace("rgb", "annotations")
                annotations_file = annotations_file[:len(annotations_file)-4] + ".json"

                instances_masks, segmentation_map, instances_labels, instances_bbox, instances_areas, \
                                 num_semantic_classes = load_image_annotations_nyu(annotations_file)

            # Not used in NYU 
            if self.args.do_kb_crop is True:
                height = image.shape[0]
                width = image.shape[1]

                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)

                image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                if self.mode == 'online_eval' and has_valid_depth:
                    depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]

            if self.mode == 'online_eval':
                sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth}
            else:
                sample = {'image': image, 'focal': focal}
        
        # Fill sample with segmentation info
        if self.args.dataset == 'nyu' and self.args.segmentation:
            sample['instances_masks'] = instances_masks
            sample['segmentation_map'] = segmentation_map
            sample['instances_labels'] = instances_labels
            sample['instances_bbox'] = instances_bbox
            sample['instances_areas'] = instances_areas
       
        # ToTensorCustom 
        if self.transform:
            sample = self.transform([sample, self.args.dataset])

        return sample


    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        
        return result


    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]

        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]

        return img, depth


    def train_preprocess(self, image, depth_gt, args, segmentation_map=None):
        """
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5 and False:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()
            if args.dataset == 'nyu' and args.segmentation:
                segmentation_map = (segmentation_map[:, ::-1, :]).copy()
        """

        # Random gamma, brightness, and color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt


    def augment_image(self, image):
        # Gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # Brightness augmentation
        if self.args.dataset == 'nyu' or self.args.dataset == 'nyud':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)

        image_aug = image_aug * brightness

        # Color augmentation - fixed
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image

        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    
    def Cut_Flip(self, image, depth):

        p = random.random()
        if p < 0.5:
            return image, depth

        image_copy = copy.deepcopy(image)
        depth_copy = copy.deepcopy(depth)
        h, w, c = image.shape

        N = 2     
        h_list = []
        h_interval_list = []   # height interval

        for i in range(N-1):
            h_list.append(random.randint(int(0.2*h), int(0.8*h)))

        h_list.append(h)
        h_list.append(0)  
        h_list.sort()
        h_list_inv = np.array([h]*(N+1))-np.array(h_list)

        for i in range(len(h_list)-1):
            h_interval_list.append(h_list[i+1]-h_list[i])

        for i in range(N):
            image[h_list[i]:h_list[i+1], :, :] = image_copy[h_list_inv[i]-h_interval_list[i]:h_list_inv[i], :, :]
            depth[h_list[i]:h_list[i+1], :, :] = depth_copy[h_list_inv[i]-h_interval_list[i]:h_list_inv[i], :, :]

        return image, depth

    
    def __len__(self):
        return len(self.filenames)


class ToTensorCustom(object):
    def __init__(self, mode, segmentation):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.max_instances = 63 
        self.segmentation = segmentation

    def __call__(self, sample_dataset):

        sample = sample_dataset[0]
        dataset = sample_dataset[1]

        image, focal = sample['image'], sample['focal']
        #image_numpy = image.copy()
        
        image = self.to_tensor(image)
        image = self.normalize(image)

        if dataset == 'kitti':
            K_p = np.array([[716.88, 0, 596.5593, 0],
                          [0, 716.88, 149.854, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=np.float32)

            inv_K_p = np.linalg.pinv(K_p)
            inv_K_p = torch.from_numpy(inv_K_p)
            
        elif dataset == 'nyu' or dataset == 'nyud':
            K_p = np.array([[518.8579, 0, 325.5824, 0],
                          [0, 518.8579, 253.7362, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=np.float32)

            inv_K_p = np.linalg.pinv(K_p)
            inv_K_p = torch.from_numpy(inv_K_p)

        if dataset == 'nyu' and self.segmentation:
            segmentation_map = torch.from_numpy(sample['segmentation_map'])
            
            # Padd to create even batch
            instances_masks = torch.stack([torch.from_numpy(arr.astype(np.float32)) for arr in sample['instances_masks']])
            num_zeros_needed = self.max_instances - instances_masks.shape[0]
            zero_tensors = [torch.zeros(1, *instances_masks.shape[1:], dtype=torch.int32) for _ in range(num_zeros_needed)]
            instances_masks = torch.cat([instances_masks] + zero_tensors, dim=0)

            """
            img = sample['instances_masks'][0].astype(np.uint8)
            colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            image_numpy[img == 1] = [255, 255, 255]
            bbox = sample['instances_bbox'][0]
            y1, x1, y2, x2 = bbox
            cv2.rectangle(image_numpy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("instance", image_numpy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """

            instances_labels = torch.from_numpy(sample['instances_labels'])
            zero_tensors = [torch.zeros(1, dtype=torch.int32) for _ in range(num_zeros_needed)]
            instances_labels = torch.cat([instances_labels] + zero_tensors, dim=0)

            instances_bbox = torch.stack([torch.from_numpy(arr) for arr in sample['instances_bbox']])
            zero_tensors = [torch.zeros((1, 4), dtype=torch.int32) for _ in range(num_zeros_needed)]
            instances_bbox = torch.cat([instances_bbox] + zero_tensors, dim=0)

            instances_areas = torch.stack([torch.from_numpy(np.asarray([arr])) for arr in sample['instances_areas']])
            zero_tensors = [torch.zeros((1, 1), dtype=torch.int32) for _ in range(num_zeros_needed)]
            instances_areas = torch.cat([instances_areas] + zero_tensors, dim=0)

            if self.mode == 'test':
                return {'image': image, 'inv_K_p': inv_K_p, 'focal': focal, 'instances_masks': instances_masks, \
                         'segmentation_map': segmentation_map, 'instances_labels': instances_labels, \
                         'instances_bbox': instances_bbox, 'instances_areas': instances_areas
                }

            depth = sample['depth']
            if self.mode == 'train':
                depth = self.to_tensor(depth)

                return {'image': image, 'depth': depth, 'focal': focal, 'instances_masks': instances_masks, \
                        'segmentation_map': segmentation_map, 'instances_labels': instances_labels, \
                        'instances_bbox': instances_bbox, 'instances_areas': instances_areas
                        }
            else:
                has_valid_depth = sample['has_valid_depth']
                return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth, \
                         'instances_masks': instances_masks, 'segmentation_map': segmentation_map, \
                         'instances_labels': instances_labels, 'instances_bbox': instances_bbox, \
                         'instances_areas': instances_areas
                       }

        else:
            if self.mode == 'test':
                return {'image': image, 'inv_K_p': inv_K_p, 'focal': focal}

            depth = sample['depth']
            if self.mode == 'train':
                depth = self.to_tensor(depth)
                return {'image': image, 'depth': depth, 'focal': focal}
            else:
                has_valid_depth = sample['has_valid_depth']
                return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth}


    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
        
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img
        
        # Handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
       
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
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
    
    num_semantic_classes = int(coco_data.get("num_categories")) + 1 # null-class ~ 14
    
    bounding_boxes = []
    areas_instances = []
    instances = []        # 0/1
    labels_instances = []
    
    segmentation_map = np.array(coco_data.get("categories"), dtype=np.int32)
    segmentation_map = create_one_hot_mask_classes_np(segmentation_map, num_semantic_classes)
    _, h, w = segmentation_map.shape # [num_classes, h, w] - 0/1

    # Iterate over annotations per image
    for ii, annotation in enumerate(coco_data.get("annotations", [])):
        
        # Unpack instance mask
        instance_pixels = np.array(annotation.get("segmentation", []), dtype=np.uint32)
        instance_map = np.zeros((h,w), dtype=np.int32)
        instance_map[instance_pixels[:, 1], instance_pixels[:, 0]] = 1
        instances.append(instance_map)
        
        # Category id
        label = annotation.get("category_id", 0)
        labels_instances.append(label)

        # Unpack bbox 
        bbox = annotation.get("bbox", [])
        bounding_boxes.append(bbox)

        # Unpack area
        area = annotation.get("area", [])
        areas_instances.append(area)
    
    return instances, segmentation_map, np.array(labels_instances), np.asarray(bounding_boxes, dtype=np.int32), \
           np.asarray(areas_instances, dtype=np.int32), num_semantic_classes


def create_one_hot_mask_classes_np(segmentation_map, num_classes):
    # segmentation_map: (h, w)
    # (classes, h, w)
    one_hot_mask = np.zeros((num_classes, *segmentation_map.shape), dtype=np.float32)
    
    """
    data = custom_cmap_instances(segmentation_map / np.max(segmentation_map))  
    data = data[:, :] * 255
    cv2.imshow("colored classes", data[:, :].astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    # Set the corresponding class index to 1
    for class_idx in range(num_classes):
        one_hot_mask[class_idx] = (segmentation_map == class_idx).astype(np.float32)
        
    return one_hot_mask
