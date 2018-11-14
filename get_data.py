#! /usr/bin/python
# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: get_data.py
Author: Yang Li
Date: 2018/11/10 17:43:31
"""

import cv2
import os
import numpy as np
import scipy.io as scio

from utils import get_patch, read_pts, logger, data_aug

# init variable
patch_size = 10
landmark_num = 68
img_size = 224
patches = [[] for i in range(landmark_num)]
labels = [[] for i in range(landmark_num)]
img_root = "/home/kb250/yl/3_graduate-design/0_DATASET"

# load data(img, bbox, pts)
data = scio.loadmat(os.path.join(img_root, 'raw_300W_release.mat'))
imgs_path = data['nameList']
bboxes = data['bbox']

# get patches
data_size = len(imgs_path)
for index in range(data_size):
    # file path building
    img_path = imgs_path[index][0][0]
    prefix = img_path.split('.')[0]
    pts_path = prefix + '.pts_occlu'
    img_path = os.path.join(img_root, img_path)
    pts_path = os.path.join(img_root, pts_path)

    # read data
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger("{} img read error".format(prefix))
        continue
    bbox = bboxes[index]
    pts_data = read_pts(pts_path, landmark_num=68)
    if pts_data is None:
        logger("{} pts parse error.".format(prefix))
        continue
    x_normalized = (pts_data[:, 0] - bbox[0]) / (bbox[1] - bbox[0])
    y_normalized = (pts_data[:, 1] - bbox[2]) / (bbox[3] - bbox[2])
    pts_data_normalized = np.stack((x_normalized, y_normalized, pts_data[:, 2]), axis=1)
    del x_normalized, y_normalized

    # data augment
    face_dups, landmark_dups, occlusion_dups = data_aug(img,
                                                        pts_data_normalized, bbox, img_size, landmark_num)
    # get patches and matched labels
    get_patch(face_dups, landmark_dups, occlusion_dups,
              patch_size, patches, labels, landmark_num)

# save patches and occlusion flag
