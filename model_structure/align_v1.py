#! /usr/bin/python
# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: align_v1.py
Author: Yang Li
Date: 2018/11/10 17:43:31
Description: Rough Face Alignment
"""
import numpy as np

from config.init_param import data_param
from model_structure.base_model import Model
from ml import classify
from utils import logger


class FaceAlignment(Model, object):
    """"Face Alignment Training"""

    def __init__(self, lr=None, epochs=None, bs=None,
                 model_name=None, classes=None, esm=None, loss=None, train_num=None):
        super(FaceAlignment, self).__init__(
            lr=lr,
            epochs=epochs,
            bs=bs,
            model_name=model_name,
            loss=loss,
            metrics=["accuracy"],
            steps_per_epochs=train_num // bs,
            classes=classes,
            esm=esm
        )

    @staticmethod
    def val_compute(imgs, labels, mean_shape=None,
                    normalizer=None, model=None, loss_compute=None):
        """Compute interocular loss of input imgs and labels"""
        loss = 0.0
        count = 0
        for img, label in zip(imgs, labels):
            prediction = FaceAlignment.test(img, mean_shape=mean_shape,
                                            normalizer=normalizer, model=model)
            loss += loss_compute(prediction, label)
            if normalizer:
                img = normalizer.transform(img)
            prediction = classify(model, img)
            loss_elem = loss_compute(prediction, label)
            loss += loss_elem
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
