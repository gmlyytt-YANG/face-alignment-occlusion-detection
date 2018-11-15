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
import pandas as pd
from sklearn import svm
from sklearn.metrics import zero_one_loss
import os

from ml import dataset_split
from utils import logger, str_or_float

# init data
landmark_size = 68
dataset_path = './data/patches_occlusion.pickle'
model_base_path = './data'
record_path = './record'
svm_config_path = './config/svm.conf'
test_size = 0.2
random_state = 0
classifiers = [svm.SVC() for _ in range(landmark_size)]
training_data = [defaultdict() for _ in range(landmark_size)]
losses = np.zeros((landmark_size,))

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
                                     max_iter=max_iter,
                                     verbose=True)
f_svm.close()

# training - fitting
logger("model fitting")
for index in range(landmark_size):
    # dataset prepare
    instances = training_data[index]['x_train']
    instances = np.reshape(instances, (len(instances), len(instances[0][0])))  # 3d -> 2d
    labels = training_data[index]['y_train']
    labels = [i * 2 - 1 for i in labels]

    # fitting
    logger("training {} models".format(index + 1))
    model_path = os.path.join(model_base_path, 'model_{}.pickle'.format(index + 1))
    if os.path.exists(model_path):
        f_pickle = open(model_path, 'rb')
        classifiers[index] = pickle.load(f_pickle)
        f_pickle.close()
    else:
        classifiers[index].fit(instances, labels)

    # validation
    instances_validation = training_data[index]['x_test']
    labels_validation = training_data[index]['y_test']
    labels_validation = [i * 2 - 1 for i in labels_validation]
    instances_validation = np.reshape(instances_validation,
                                      (len(instances_validation), len(instances_validation[0][0])))
    labels_validation_predict = classifiers[index].predict(instances_validation)
    logger('saving prediction result')
    df = pd.DataFrame({'labels': labels_validation, 'predict': labels_validation_predict})
    df.to_csv(os.path.join(record_path, '{}_labels_compare.csv'.format(index + 1)))

    # compute loss
    zo_loss = zero_one_loss(labels_validation, labels_validation_predict)
    losses[index] = zo_loss
    logger("the validation loss of {} models is {}".format(index + 1, zo_loss))

    logger("-------------------------------------------")

    # save model
    if os.path.exists(model_path):
        continue
    f_pickle = open(model_path, 'wb')
    pickle.dump(classifiers[index], f_pickle)
    f_pickle.close()

# save loss
logger('saving loss result')
df = pd.DataFrame({'loss': losses})
df.to_csv(os.path.join(record_path, 'loss.csv'))
