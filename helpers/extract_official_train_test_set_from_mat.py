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
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# Define a custom colormap using LinearSegmentedColormap
colors = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]  # Replace with your desired colors
n_bins = [894]  # Number of bins, can be adjusted based on your data

cmap_name = "custom_colormap"
custom_cmap_labels = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins[0])
custom_cmap_instances = LinearSegmentedColormap.from_list(cmap_name, colors, N=37)


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

def convert_image(i, scene, depth_raw, image):

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


def convert_instance(i, scene, image):

    idx = int(i) + 1
    if idx in train_images:
        train_test = "train"
    else:
        assert idx in test_images, "index %d neither found in training set nor in test set" % idx
        train_test = "test"

    folder = "%s/%s/%s" % (out_folder, train_test, scene)
    if not os.path.exists(folder):
        os.makedirs(folder)

    image = image[:, ::-1]

    # Filter #
    h,w = image.shape
    min_instance_area = 0.25/100 * h * w
    # Specify the threshold
    # Count the occurrences of each ID
    id_counts = np.bincount(image.flatten())
    # Identify IDs below the threshold and set them to 0
    below_threshold_ids = np.where(id_counts < min_instance_area)[0]

    # Update the original data array
    for id_to_zero in below_threshold_ids:
        image[image == id_to_zero] = 0

    image_black_boundary = np.zeros((480, 640), dtype=np.uint8)
    image_black_boundary[7:474, 7:632] = image[7:474, 7:632]
    #image_black_boundary = cv2.flip(image_black_boundary, 1)
    cv2.imwrite("%s/instance_%05d.jpg" % (folder, i), image_black_boundary)

    image = image_black_boundary

    unique_ids = np.unique(image)
    unique_ids = unique_ids[unique_ids != 0]
    bounding_boxes = {}

    for id_value in unique_ids:
        bounding_box = encapsulate_bounding_boxes(image, id_value)
        bounding_boxes[id_value] = bounding_box

    print(bounding_boxes)

    # Create a dictionary to store the separate maps for each ID
    id_maps = {}

    # Iterate over each unique ID
    for id_value in unique_ids:
        print(id_value)
        # Create a new map with zeros in areas where the ID is not present
        id_maps[id_value] = np.where(image == id_value, 255, 0)

    # Print the resulting maps
    for id_value, map_data in id_maps.items():
        print(f'Map for ID {id_value}:')
        print(map_data)
        print()
        cv2.imshow('image', map_data.astype(np.uint8))
        cv2.waitKey(0)


    """
    data_dict = {
        'image_name': 'your_image_name.jpg',  # Replace with the actual image name
        'bounding_boxes': {},
        'maps': {}
    }
    # Store bounding boxes
    for id_value in unique_ids:
        bounding_box = encapsulate_bounding_boxes(image, id_value)
        data_dict['bounding_boxes'][id_value] = bounding_box

    # Store maps
    for id_value in unique_ids:
        map_data = np.where(image == id_value, 1, 0).tolist()
        data_dict['maps'][id_value] = map_data

    # Save to JSON file
    output_file_path = 'output_data.json'
    with open(output_file_path, 'w') as json_file:
        json.dump(data_dict, json_file)
    """

    image = image_black_boundary
    # Map data values to RGB colors using the colormap
    rgb_data = custom_cmap_instances(image / np.max(image))  # Normalize the data to [0, 1]
    rgb_data = rgb_data[:, :, :3] * 255
    # Visualize bounding boxes on the image
    for id, bounding_box in bounding_boxes.items():
        y1, x1, y2, x2 = bounding_box
        cv2.rectangle(rgb_data, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save the colored image
    cv2.imwrite("%s/instance_colored_%05d.jpg" % (folder, i), (rgb_data).astype(np.uint8))
    print(folder, i)
    #exit()




def convert_labels(i, scene, image):

    idx = int(i) + 1
    if idx in train_images:
        train_test = "train"
    else:
        assert idx in test_images, "index %d neither found in training set nor in test set" % idx
        train_test = "test"

    folder = "%s/%s/%s" % (out_folder, train_test, scene)
    if not os.path.exists(folder):
        os.makedirs(folder)
    image = image[:, ::-1]
    image_black_boundary = np.zeros((480, 640), dtype=np.uint8)
    image_black_boundary[7:474, 7:632] = image[7:474, 7:632]
    image_black_boundary = cv2.flip(image_black_boundary, 1)
    cv2.imwrite("%s/label_%05d.jpg" % (folder, i), image_black_boundary)
    image = image_black_boundary

    unique_ids = np.unique(image)
    print(unique_ids)
    exit()
    # Map data values to RGB colors using the colormap
    rgb_data = custom_cmap_labels(image / np.max(image))  # Normalize the data to [0, 1]

    # Save the colored image
    cv2.imwrite("%s/label_colored_%05d.jpg" % (folder, i), (rgb_data[:, :, :3] * 255).astype(np.uint8))

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

    labels = h5_file['labels']
    print("processing labels")
    for i, image in enumerate(labels):
        print("image", i + 1, "/", len(labels))
        convert_labels(i, scenes[i],  image.T)
        break
    instances = h5_file['instances']

    print("processing instances")
    for i, image in enumerate(instances):
        print("image", i + 1, "/", len(instances))
        convert_instance(i, scenes[i],  image.T)
        break


    depth_raw = h5_file['rawDepths']

    print("reading", sys.argv[1])

    images = h5_file['images']

    print("processing images")
    for i, image in enumerate(images):
        print("image", i + 1, "/", len(images))
        convert_image(i, scenes[i], depth_raw[i, :, :].T, image.T)

    print("Finished")
