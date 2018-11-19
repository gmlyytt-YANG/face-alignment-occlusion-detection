#! /usr/bin/python
# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: init_param.py
Author: Yang Li
Date: 2018/11/19 20:36:31
Description: param config
"""

import os

param = {
    'print_debug': True,

    # path-related
    'root_dir': '../',
    'img_root_dir': os.path.join(os.path.abspath('..'), "0_DATASET/origin_img"),
    'data_save_dir': os.path.join(os.getcwd(), "data"),
    'model_dir': './model',

    # dataset related
    'landmark_num': 68,
    'img_size': 128,
    'channel': 3,
    'test_size': 0.3,  # split test size
    'random_state': 0,  # split random state

    # dl-related
    'epochs': 75,
    'init_lr': 1e-3,
    'bs': 32
}
