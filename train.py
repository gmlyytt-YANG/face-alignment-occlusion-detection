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
from prepare.utils import logger
from prepare.data_gen import train_data_feed, validation_data_feed

# loading data
logger("loading data")
train_dir = os.path.join(param['data_save_dir'], "train")
validation_dir = os.path.join(param['data_save_dir'], "validation")
validation_data, validation_labels = validation_data_feed(validation_dir, print_debug=param['print_debug'])

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
    train_data_feed(param['bs'], train_dir),
    validation_data=(validation_data, validation_labels),
    steps_per_epoch=len(os.listdir(train_dir)) // param['bs'],
    epochs=param['epochs'], verbose=1)

# save model
logger("saving model")
if not os.path.exists(param['model_dir']):
    os.mkdir(param['model_dir'])
model.save(os.path.join(param['model_dir'], param['model_name']))
