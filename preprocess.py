#! /usr/bin/python
# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: preprocess.py
Author: Yang Li
Date: 2018/11/28 09:31:31
Description: Program Main Entry
"""
import os

from config.init_param import data_param
from prepare.img_server import ImageServer
from utils import load_basic_info

# load data(img, bbox, pts)
mat_file = os.path.join(data_param['img_root_dir'], 'raw_300W_release.mat')
img_paths, bboxes = load_basic_info(mat_file, img_root=data_param['img_root_dir'])
img_server = ImageServer(img_size=data_param['img_size'],
                         color=True if data_param['channel'] == 3 else False,
                         print_debug=True)
# process 
chosen_indices = range(3148)
img_server.process(img_paths, bboxes, chosen_indices)
