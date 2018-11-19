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
import numpy as np
import os
import pickle

from keras.optimizers import Adam

from config.init_param import param
from model_structure.model import SmallerVGGNet
from prepare.utils import generate_batch_data_random, logger


# load data
def load_data(mode, print_debug=False):
    data_list = []
    labels_list = []
    root_path = os.path.join(param['data_save_dir'], mode)
    count = 0
    for path in os.listdir(root_path):
        data_path = os.path.join(root_path, path)
        f_data = open(data_path, 'rb')
        data = pickle.load(f_data)
        data_list.append(data['image'])
        labels_list.append(data['label'])
        f_data.close()
        if print_debug:
            if (count + 1) % 500 == 0:
                logger("loaded {} data in phase {}".format(count + 1, mode))
        count = count + 1
        # if count > 1000:
        #     break
    return np.array(data_list), np.array(labels_list)


# loading data
logger("loading data")
train_data, train_labels = load_data("train", print_debug=param['print_debug'])
validation_data, validation_labels = load_data("validation",
                                               print_debug=param['print_debug'])

# build model
logger("building model")
model = SmallerVGGNet.build(
    width=param['img_size'], height=param['img_size'],
    depth=param['channel'], classes=param['landmark_num'],
    final_act="sigmoid")
opt = Adam(lr=param['init_lr'], decay=param['init_lr'] / param['epochs'])
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train model
logger("training")
H = model.fit_generator(
    generate_batch_data_random(train_data, train_labels, batch_size=param['bs']),
    validation_data=(validation_data, validation_labels),
    steps_per_epoch=len(train_data) // param['bs'],
    epochs=param['epochs'], verbose=1)

# save model
logger("saving model")
if not os.path.exists(param['model_dir']):
    os.mkdir(param['model_dir'])
model.save(param['model_dir'])
