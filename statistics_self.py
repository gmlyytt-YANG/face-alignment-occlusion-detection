#! /usr/bin/python
# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: stat.py
Author: Yang Li
Date: 2018/11/15 09:23:31
Description: Basic Statistics
"""

import numpy as np
import pickle
import os

from utils import logger

# init data
landmark_size = 68
dataset_path = './data/patches_occlusion.pickle'

# load data
logger("loading data")
if os.path.exists(dataset_path):
    with open(dataset_path, 'rb') as f_pickle:
        data = pickle.load(f_pickle)
    labels = data['occlusion']
    assert len(labels) == landmark_size
else:
    logger("missing dataset")

# stat
labels_ratio = np.zeros((landmark_size,))

for index in range(landmark_size):
    den = len(labels[index])
    labels[index] = np.reshape(labels[index], (1, den))
    # labels[index] = [int(_) for _ in labels[index]]
    num = np.sum(labels[index])
    labels_ratio[index] = float(num) / den
    print(float(num) / den)

# plot
