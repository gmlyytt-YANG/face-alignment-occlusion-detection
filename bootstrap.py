#! /usr/bin/python
# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: bootstrap.py
Author: Yang Li
Date: 2018/11/21 09:41:31
Description: Program Main Entry
"""
import argparse
import os

from config.init_param import occlu_param
from occlusion_detection import OcclusionDetection
from prepare.utils import load_imgs

# load parameter
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epoch", type=int, default=75,
                help="epochs of training")
ap.add_argument("-bs", "--batch_size", type=int, default=32,
                help="batch size of training")
ap.add_argument("-lr", "--init_lr", type=float, default=1e-3,
                help="learning rate")
args = vars(ap.parse_args())

occlu_param['epochs'] = args['epoch']
occlu_param['bs'] = args['batch_size']
occlu_param['init_lr'] = args['init_lr']

# occlusion detection
OcclusionDetection.data_pre()
# OcclusionDetection.train()

mat_file = os.path.join(occlu_param['img_root_dir'], 'raw_300W_release.mat')
for face in load_imgs(occlu_param['img_root_dir'],
                      mat_file_name=mat_file,
                      total=False,
                      chosed=[6, 7]):
    OcclusionDetection.classify(face, need_to_normalize=True)
