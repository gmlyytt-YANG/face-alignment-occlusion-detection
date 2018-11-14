#! /usr/bin/python
# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: occlusion_detection.py
Author: Yang Li
Date: 2018/11/14 14:49:31
Description: Occlusion Detection
"""

from collections import defaultdict
import numpy as np
import pickle
from sklearn import svm
import os

from ml import dataset_split
from utils import logger, str_or_float

# init data
landmark_size = 68
dataset_path = './data/patches_occlusion.pickle'
svm_config_path = './config/svm.conf'
test_size = 0.2
random_state = 0
classifiers = [svm.SVC() for _ in range(landmark_size)]
training_data = [defaultdict() for _ in range(landmark_size)]

# load data
logger("loading data")
if os.path.exists(dataset_path):
    with open(dataset_path, 'rb') as f_pickle:
        data = pickle.load(f_pickle)
    patches = data['patches']
    labels = data['occlusion']
    assert len(patches) == landmark_size
    assert len(labels) == landmark_size
else:
    logger("missing dataset")

# data splitting for each classifier
logger("data splitting")
for index in range(landmark_size):
    training_data[index] = dataset_split(patches[index], labels[index],
                                         test_size, random_state)

# training - load config
logger("loading config")
f_svm = open(svm_config_path, 'r')
count = 0
for line in f_svm.readlines():
    count += 1
    if count > landmark_size:
        break
    if count == 1:  # first row is introduction
        continue
    config = line.split(' ')
    C = float(config[0])
    kernel = config[1]
    gamma = str_or_float(config[2])
    coef0 = float(config[3])
    tol = float(config[4])
    max_iter = int(config[5].strip())
    classifiers[count - 1] = svm.SVC(C=C,
                                     kernel=kernel,
                                     gamma=gamma,
                                     coef0=coef0,
                                     tol=tol,
                                     max_iter=max_iter)
f_svm.close()

# training - fitting
logger("model fitting")
for index in range(landmark_size):
    instances = training_data[index]['x_train']
    instances = np.reshape(instances, (len(instances), len(instances[0][0])))  # 3d -> 2d
    labels = training_data[index]['y_train']
    classifiers[index].fit(instances, labels)
    logger("trained {} models".format(index + 1))