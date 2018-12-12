#! /usr/bin/python
# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: parse_param.py
Author: Yang Li
Date: 2018/12/11 21:28:31
Description: param parse
"""
import os
from keras.models import load_model

from model_structure.vgg16 import Vgg16CutFC2, Vgg16Regress
from model_structure.resnet import ResNet
from ml import landmark_loss
from ml import landmark_loss_compute
from ml import landmark_delta_loss
from ml import landmark_delta_loss_compute
from config.init_param import data_param


def parse_param(model_type, loss_name):
    if model_type == 'vgg16_clf':
        model_structure = Vgg16CutFC2()
    elif model_type == 'vgg16_rgr':
        model_structure = Vgg16Regress()
    elif model_type == 'resnet_rgr':
        model_structure = ResNet()
    else:
        raise ValueError('there is no such model structure!')
    if loss_name == 'landmark_loss':
        loss = landmark_loss
        loss_compute = landmark_loss_compute
    elif loss_name == 'landmark_delta_loss':
        loss = landmark_delta_loss
        loss_compute = landmark_delta_loss_compute
    else:
        loss = None
        loss_compute = None
    return model_structure, loss, loss_compute
