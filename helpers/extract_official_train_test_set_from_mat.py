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
# Define a custom colormap using LinearSegmentedColormap
colors = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]  # Replace with your desired colors
n_bins = [894]  # Number of bins, can be adjusted based on your data

cmap_name = "custom_colormap"
custom_cmap_labels = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins[0])
custom_cmap_instances = LinearSegmentedColormap.from_list(cmap_name, colors, N=37)

def get_instance_masks(imgObjectLabels, imgInstances):
    H, W = imgObjectLabels.shape

    pairs = np.unique(np.column_stack((imgObjectLabels.flatten(), np.uint16(imgInstances.flatten()))), axis=0)
    pairs = pairs[pairs.sum(axis=1) != 0]

    N = pairs.shape[0]

    instanceMasks = []
    instanceLabels = []
    boxes = []
    areas = []
    pixel_coordinates = []

    min_instance_area = 0#0.25/100 * H * W
    for ii in range(N):
        tmp = np.logical_and(imgObjectLabels == pairs[ii, 0], imgInstances == pairs[ii, 1])
        pixels = np.sum(tmp)
        if pixels > min_instance_area:
            areas.append(pixels)
            instanceMasks.append(np.uint8(tmp) * 255)
            instanceLabels.append(pairs[ii, 0])

            # Find bounding box coordinates
            nonzero_rows, nonzero_cols = np.nonzero(tmp)
            bbox = [np.min(nonzero_rows), np.min(nonzero_cols), np.max(nonzero_rows), np.max(nonzero_cols)]
            offset = 2
            bbox[0] = max(bbox[0] - offset, 0)
            bbox[1] = max(bbox[1] - offset, 0)
            bbox[2] = min(bbox[2] + offset, labels.shape[1] - 1)
            bbox[3] = min(bbox[3] + offset, labels.shape[0] - 1)
            boxes.append(bbox)
            pixel_coordinates.append(np.column_stack((nonzero_cols, nonzero_rows)))

        else:
            print("instance filterted with: ", str(pixels), " pixels")
    return np.stack(instanceMasks, axis=2), np.array(instanceLabels), np.array(boxes), np.array(areas), pixel_coordinates


def encapsulate_bounding_boxes(map_data, id_value):
    # Find the locations where the ID occurs
    locations = np.where(map_data == id_value)

    if len(locations[0]) == 0:
        # ID not found in the map
        return None

    # Calculate bounding box coordinates
    y1 = np.min(locations[0])
    x1 = np.min(locations[1])
    y2 = np.max(locations[0]) + 1
    x2 = np.max(locations[1]) + 1

    return [y1, x1, y2, x2]

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
    print("%s/sync_depth_%05d.png" % (folder, i))
    image = image[:, :, ::-1]
    image_black_boundary = np.zeros((480, 640, 3), dtype=np.uint8)
    image_black_boundary[7:474, 7:632, :] = image[7:474, 7:632, :]
    cv2.imwrite("%s/rgb_%05d.jpg" % (folder, i), image_black_boundary)


def convert_instances_and_semantic_mask(i, scene, image, label_map):
    instances, labels, boxes, areas, pixel_coordinates = get_instance_masks(label_map, image)
    #for i in range(31):
    #    cv2.imshow(str(i), instances[:, :, i].astype(np.uint8))
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()

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

    labels = labels.tolist()
    # Add category information (assuming you have a fixed set of semantic labels)
    coco_data["categories"] = label_map.tolist()
    # Add instance annotations
    annotation_id = 1
    for ii in range(instances.shape[2]):
        bbox = boxes[ii]
        segmentation = pixel_coordinates[ii].tolist() #instances[:, :, ii].tolist()  # Convert to list for JSON serialization
        annotation = {
            "id": annotation_id,
            "category_id": labels[ii],
            "segmentation": segmentation,
            "bbox": bbox.tolist(),
            "area": areas[ii].tolist()
        }

        coco_data["annotations"].append(annotation)
        annotation_id += 1

    annotations_name = "%s/annotations_%05d.json" % (folder, i)
    print(annotations_name)
    # Save the data to a JSON file
    with open(annotations_name, "w") as json_file:
        json.dump(coco_data, json_file)

    rgb_data = custom_cmap_instances(label_map / np.max(label_map))  # Normalize the data to [0, 1]
    rgb_data = rgb_data[:, :] * 255
    # Visualize bounding boxes on the image
    for box in boxes:
        y1, x1, y2, x2 = box
        cv2.rectangle(rgb_data, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite("%s/map_%05d.png" % (folder, i), rgb_data)
    """
    cv2.imshow("i", rgb_data[:, :].astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    segmentations, labels_map_load, labels_load, bounding_boxes, areas_load = load_json_and_unpack(annotations_name)
    print(np.array_equal(labels, labels_load))
    print(np.array_equal(labels_map_load, label_map))
    print(np.array_equal(segmentations[0], pixel_coordinates[0]))
    print(np.array_equal(areas, areas_load))
    print(np.array_equal(boxes, bounding_boxes))
    exit()
    """
    #rgb_data = custom_cmap_instances(label_map / np.max(label_map))  # Normalize the data to [0, 1]
    #rgb_data = rgb_data[:, :] * 255
    # Visualize bounding boxes on the image
    #for box in boxes:
    #    y1, x1, y2, x2 = box
    #    cv2.rectangle(rgb_data, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #cv2.imshow("j", rgb_data[:, :].astype(np.uint8))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #for i in range(31):
    #    cv2.imshow(str(i), instances[:, :, i].astype(np.uint8))
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()
    #exit()

def load_json_and_unpack(json_file_path):
    with open(json_file_path, 'r') as file:
        coco_data = json.load(file)

    # Initialize empty arrays
    bounding_boxes_list = []
    areas_list = []
    segmentations = []
    labels_map = np.array(coco_data.get("categories"))
    labels = []
    # Iterate over annotations in the COCO-like format
    for annotation in coco_data.get("annotations", []):
        # Unpack segmentation (assuming it's a list)
        segmentation = np.array(annotation.get("segmentation", []), dtype=np.uint16)
        segmentations.append(segmentation)
        # Unpack category_id, assuming it's the label in your semantic map
        label = annotation.get("category_id", 0)
        labels.append(label)

        # Unpack bbox (bounding box) as a list
        bbox = annotation.get("bbox", [])
        bounding_boxes_list.append(bbox)

        # Unpack area
        area = annotation.get("area", 0)
        areas_list.append(area)

    return segmentations, labels_map, labels, np.asarray(bounding_boxes_list), np.asarray(areas_list)


def convert_names(names):
    counter = 1
    data = {}
    for name in names:
        data[name] = counter
        counter += 1

    json_data = json.dumps(data, indent=2)
    print(json_data)
    # Specify the file path
    file_path = out_folder + '/labels.json'

    # Write to the file
    with open(file_path, 'w') as file:
        file.write(json_data)

if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("usage: %s <h5_file> <train_test_split> <out_folder>" % sys.argv[0], file=sys.stderr)
        sys.exit(0)

    h5_file = h5py.File(sys.argv[1], "r")
    # h5py is not able to open that file. but scipy is
    train_test = scipy.io.loadmat(sys.argv[2])
    out_folder = sys.argv[3]

    test_images = set([int(x) for x in train_test["testNdxs"]])
    train_images = set([int(x) for x in train_test["trainNdxs"]])
    print("%d training images" % len(train_images))
    print("%d test images" % len(test_images))
    scenes = [u''.join(chr(c[0]) for c in h5_file[obj_ref]) for obj_ref in h5_file['sceneTypes'][0]]

    names = h5_file['names']
    names = [u''.join(chr(c[0]) for c in h5_file[obj_ref]) for obj_ref in names[0]]
    convert_names(names)


    print("reading", sys.argv[1])
    labels = h5_file['labels']
    images = h5_file['images']
    instances = h5_file['instances']
    depth_raw = h5_file['rawDepths']

    print("processing instances")
    for i, (instance, label, image) in enumerate(zip(instances, labels, images)):
        print("image", i + 1, "/", len(instances))
        convert_instances_and_semantic_mask(i, scenes[i],  instance.T, label.T)
        convert_image_and_depth(i, scenes[i], depth_raw[i, :, :].T, image.T)
    print("Finished")
