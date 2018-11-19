#! /usr/bin/python
# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: data_prepare.py
Author: Yang Li
Date: 2018/11/10 17:43:31
Description: Data Augment
"""

import os
import scipy.io as scio

from config.init_param import param
from prepare.ImageServer import ImageServer
from prepare.utils import logger

# load data(img, bbox, pts)
data = scio.loadmat(os.path.join(param['img_root_dir'], 'raw_300W_release.mat'))
img_paths = data['nameList']
img_paths = [i[0][0] for i in img_paths]
bboxes = data['bbox']

# data prepare
img_server = ImageServer(data_size=len(img_paths),
                         img_size=param['img_size'], color=True)
img_server.process(img_root=param['img_root_dir'], img_paths=img_paths,
                   bounding_boxes=bboxes, print_debug=param['print_debug'])

# splitting
logger("train validation splitting")
img_server.train_validation_split(test_size=param['test_size'],
                                  random_state=param['random_state'])

# saving
logger("saving data")
img_server.save(param['data_save_dir'], print_debug=param['print_debug'])
