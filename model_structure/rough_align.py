#! /usr/bin/python
# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: rough_align.py
Author: Yang Li
Date: 2018/11/10 17:43:31
Description: Rough Face Alignment
"""
import numpy as np
import os

from config.init_param import data_param, face_alignment_rough_param
from model_structure.base_model import Model
from ml import classify
from ml import landmark_loss_compute
from utils import logger


class FaceAlignment(Model, object):
    """"Face Alignment Training"""
    def __init__(self, loss):
        train_dir = os.path.join(data_param['data_save_dir'], 'train')
        super(FaceAlignment, self).__init__(
            lr=face_alignment_rough_param['init_lr'],
            epochs=face_alignment_rough_param['epochs'],
            bs=face_alignment_rough_param['bs'],
            model_name=face_alignment_rough_param['model_name'],
            loss=loss,
            metrics=["accuracy"],
            steps_per_epochs=len(os.listdir(train_dir)) // (face_alignment_rough_param['bs'] * 6),
            classes=data_param['landmark_num'] * 2
        )

    @staticmethod
    def val_compute(imgs, labels, normalizer=None, model=None):
        """Compute interocular loss of input imgs and labels"""
        loss = 0.0
        count = 0
        for img, label in zip(imgs, labels):
            if normalizer:
                img = normalizer.transform(img)
            prediction = classify(model, img)
            loss += landmark_loss_compute(prediction, label)
            count += 1
            if data_param['print_debug'] and count % 100 == 0:
                logger("predicted {} imgs".format(count))
        logger("test loss is {}".format(loss / count))

    @staticmethod
    def test(img, mean_shape=None, normalizer=None, model=None):
        """Compute prediction of input img"""
        if normalizer is not None:
            img = normalizer.transform(img)
        prediction = classify(model, img)
        if mean_shape is not None:
            prediction = np.reshape(prediction, (data_param['landmark_num'], 2)) + mean_shape
        return prediction
