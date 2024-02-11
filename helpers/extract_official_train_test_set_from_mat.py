#!/usr/bin/env python
# -*- coding: utf-8 -*-
#######################################################################################
# The MIT License

# Copyright (c) 2014       Hannes Schulz, University of Bonn  <schulz@ais.uni-bonn.de>
# Copyright (c) 2013       Benedikt Waldvogel, University of Bonn <mail@bwaldvogel.de>
# Copyright (c) 2008-2009  Sebastian Nowozin                       <nowozin@gmail.com>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#######################################################################################
#
# Helper script to convert the NYU Depth v2 dataset Matlab file into a set of
# PNG and JPEG images.
#
# See https://github.com/deeplearningais/curfil/wiki/Training-and-Prediction-with-the-NYU-Depth-v2-Dataset

from __future__ import print_function
from scipy.io import loadmat
import h5py
import numpy as np
import os
import scipy.io
import sys
import cv2
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from utils import *

colors = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]  
n_bins = [894]  

cmap_name = "custom_colormap"
custom_cmap_labels = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins[0])
custom_cmap_instances = LinearSegmentedColormap.from_list(cmap_name, colors, N=37)

CLASSES_13_FILEPATH = '/usr/stud/petp/storage/user/petp/datasets/bts/utils/class13Mapping.mat'
CLASSES_40_FILEPATH = '/usr/stud/petp/storage/user/petp/datasets/bts/utils/classMapping40.mat'                                                     
                                                   
def get_instances_masks(labels_map, instances_map):

    instance_masks = []
    instance_lables = []
    boxes = []
    areas = []
    pixel_coordinates = []

    pairs = np.unique(np.column_stack((labels_map.flatten(), np.uint32(instances_map.flatten()))), axis=0)
    pairs = pairs[pairs.sum(axis=1) != 0]

    N = pairs.shape[0]
    H, W = labels_map.shape

    min_instance_area = 0.0025 * H * W
    offset = 2

    for ii in range(N):
        tmp = np.logical_and(labels_map == pairs[ii, 0], instances_map == pairs[ii, 1])
        pixels = np.sum(tmp)
        
        if pixels > min_instance_area:
            instance_masks.append(tmp)
            instance_lables.append(pairs[ii, 0])
            areas.append(pixels)
       
            # Find bounding box coordinates
            nonzero_rows, nonzero_cols = np.nonzero(tmp)
            pixel_coordinates.append(np.column_stack((nonzero_cols, nonzero_rows)))
            
            bbox = [np.min(nonzero_rows), np.min(nonzero_cols), np.max(nonzero_rows), np.max(nonzero_cols)]
    
            bbox[0] = max(bbox[0] - offset, 0)
            bbox[1] = max(bbox[1] - offset, 0)
            bbox[2] = min(bbox[2] + offset, labels.shape[1] - 1)
            bbox[3] = min(bbox[3] + offset, labels.shape[0] - 1)
            boxes.append(bbox)

        else:
            #print("Instance filterted with: ", str(pixels), " pixels")
            pass
    
    return np.stack(instance_masks, axis=2), np.array(instance_lables, dtype=np.int32), np.array(boxes, dtype=np.int32), np.array(areas, dtype=np.float32), pixel_coordinates


def convert_image_and_depth(i, scene, depth_raw, image):

    idx = int(i) + 1
    if idx in train_images:
        train_test = "train"
    else:
        assert idx in test_images, "index %d neither found in training set nor in test set" % idx
        train_test = "test"

    folder = "%s/%s/%s" % (out_folder, train_test, scene)
    if not os.path.exists(folder):
        os.makedirs(folder)

    img_depth = depth_raw * 1000.0
    img_depth_uint16 = img_depth.astype(np.uint16)
    cv2.imwrite("%s/sync_depth_%05d.png" % (folder, i), img_depth_uint16)
    
    image = image[:, :, ::-1]
    
    image_black_boundary = np.zeros((480, 640, 3), dtype=np.uint8)
    image_black_boundary[7:474, 7:632, :] = image[7:474, 7:632, :]
    cv2.imwrite("%s/rgb_%05d.jpg" % (folder, i), image_black_boundary)
    
    print("%s/sync_depth_%05d.png" % (folder, i))


def convert_instances_and_semantic_mask(i, scene, instances_input, label_map, image, mapping_894_to_40, mapping_40_to_13, mapping_13_to_5):
    
    # Instances #
    instances, labels, boxes, areas, pixel_coordinates = get_instances_masks(label_map, instances_input)
    label_map = mapping_894_to_40[label_map]
    label_map = mapping_40_to_13[label_map]

    # Semantic #
    labels = mapping_894_to_40[labels]
    labels = mapping_40_to_13[labels]

    # For debug #
    image = image[:, :, ::-1]
    image_black = np.zeros((480, 640, 3), dtype=np.uint8)
    image_black[7:474, 7:632, :] = image[7:474, 7:632, :]

    idx = int(i) + 1
    if idx in train_images:
        train_test = "train"
    else:
        assert idx in test_images, "index %d neither found in training set nor in test set" % idx
        train_test = "test"

    folder = "%s/%s/%s" % (out_folder, train_test, scene)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Create a COCO-like dictionary
    coco_data = {
        "annotations": [],
        "categories": []
    }

    # Semantics #
    labels = labels.tolist()
    coco_data["categories"] = label_map.tolist()
    coco_data["num_categories"] = 13
    
    # Add instance annotations
    annotation_id = 1
    for ii in range(instances.shape[2]):
        segmentation = pixel_coordinates[ii].tolist() 
        
        #segmentation = np.array(segmentation, dtype=np.int32)
        #segmentation_map = np.zeros((480, 640), dtype=np.int32)
        #image_tmp = image_black.copy()
        #image_tmp[segmentation[:, 1], segmentation[:, 0]] = 0
        #cv2.imshow(str(ii), image_tmp[:, :].astype(np.uint8))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        annotation = {
            "id": annotation_id,
            "category_id": labels[ii],
            "segmentation": segmentation,
            "bbox": bbox[ii].tolist(),
            "area": areas[ii].tolist()
        }

        coco_data["annotations"].append(annotation)
        annotation_id += 1

    annotations_name = "%s/annotations_%05d.json" % (folder, i)
    print(annotations_name)

    # Save the data to a JSON file
    with open(annotations_name, "w") as json_file:
        json.dump(coco_data, json_file)

    # Viz # 
    """
    rgb_data = custom_cmap_instances(label_map / np.max(label_map))  # Normalize the data to [0, 1]
    rgb_data = rgb_data[:, :] * 255
    for box in boxes:
        y1, x1, y2, x2 = box
        cv2.rectangle(rgb_data, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("i", rgb_data[:, :].astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """


def load_json_and_unpack(json_file_path):
    with open(json_file_path, 'r') as file:
        coco_data = json.load(file)

    bounding_boxes_list = []
    areas_list = []
    segmentations = []
    labels_map = np.array(coco_data.get("categories"))
    labels = []
    
    # Iterate over annotations in the COCO-like format
    for annotation in coco_data.get("annotations", []):
        segmentation = np.array(annotation.get("segmentation", []), dtype=np.uint16)
        segmentations.append(segmentation)
        label = annotation.get("category_id", 0)
        labels.append(label)

        bbox = annotation.get("bbox", [])
        bounding_boxes_list.append(bbox)

        area = annotation.get("area", 0)
        areas_list.append(area)

    return segmentations, labels_map, labels, np.asarray(bounding_boxes_list), np.asarray(areas_list)


def write_labels_names(names):
    counter = 1
    data = {}
    for name in names:
        data[name] = counter
        counter += 1

    data = {}
    for idx, value in enumerate(SEMANTIC_LABEL_LIST_13._class_names):
        data[value] = idx

    json_data = json.dumps(data, indent=2)
    print(json_data)

    file_path = out_folder + '/labels.json'
    with open(file_path, 'w') as file:
        file.write(json_data)

if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("usage: %s <h5_file> <train_test_split> <out_folder>" % sys.argv[0], file=sys.stderr)
        sys.exit(0)

    classes_40 = loadmat(CLASSES_40_FILEPATH)
    classes_13 = loadmat(CLASSES_13_FILEPATH)['classMapping13'][0][0]

    mapping_894_to_40 = np.concatenate([[0], classes_40['mapClass'][0]])
    mapping_40_to_13 = np.concatenate([[0], classes_13[0][0]])
    mapping_13_to_5 = np.asarray([0, 1, 1, 4, 1, 3, 1, 1, 1, 1, 1, 1, 2, 1])

    h5_file = h5py.File(sys.argv[1], "r")
    train_test = scipy.io.loadmat(sys.argv[2])
    out_folder = sys.argv[3]

    test_images = set([int(x) for x in train_test["testNdxs"]])
    train_images = set([int(x) for x in train_test["trainNdxs"]])
    print("%d training images" % len(train_images))
    print("%d test images" % len(test_images))
    scenes = [u''.join(chr(c[0]) for c in h5_file[obj_ref]) for obj_ref in h5_file['sceneTypes'][0]]

    names = h5_file['names']
    names = [u''.join(chr(c[0]) for c in h5_file[obj_ref]) for obj_ref in names[0]]
    write_labels_names(names)

    print("reading", sys.argv[1])
    labels = h5_file['labels']
    images = h5_file['images']
    instances = h5_file['instances']
    depth_raw = h5_file['rawDepths']

    for i, (instance, label, image) in enumerate(zip(instances, labels, images)):
        convert_instances_and_semantic_mask(i, scenes[i],  instance.T, label.T, image.T, mapping_894_to_40, mapping_40_to_13, mapping_13_to_5)
        convert_image_and_depth(i, scenes[i], depth_raw[i, :, :].T, image.T)
    print("Finished")
