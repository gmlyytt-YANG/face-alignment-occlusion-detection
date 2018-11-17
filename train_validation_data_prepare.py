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
Description: Data Augment
"""

import os
import scipy.io as scio

from prepare.ImageServer import ImageServer
from prepare.utils import logger

# init variable
print_debug = True
landmark_num = 68
img_size = 512
test_size = 0.3
random_state = 0
dataset_save_path = "/home/kb250/yl/3_graduate-design/6_YL_18_FACEALIGNMENT/data"
img_root = "/home/kb250/yl/3_graduate-design/0_DATASET/origin_img"

# load data(img, bbox, pts)
data = scio.loadmat(os.path.join(img_root, 'raw_300W_release.mat'))
img_paths = data['nameList']
img_paths = [i[0][0] for i in img_paths]
bboxes = data['bbox']

# data prepare
img_server = ImageServer(data_size=len(img_paths),
                         img_size=img_size, color=True)
logger("preparing data")
img_server.prepare_data(img_root=img_root, img_paths=img_paths,
                        bounding_boxes=bboxes, print_debug=print_debug)
logger("loading imgs")
img_server.load_imgs(print_debug=print_debug)
logger("saving data")
img_server.save(dataset_save_path)