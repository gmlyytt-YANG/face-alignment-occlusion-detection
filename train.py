#! /usr/bin/python
# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: train.py
Author: Yang Li
Date: 2018/11/19 11:28:31
Description: Model Train
"""
import os
import sys

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer

# init param
EPOCHS = 75
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (128, 128, 3)

# init data
root_dir = os.getcwd()
data_source_path = os.path.join(root_dir, "data")

data = []
labels = []

# load data
