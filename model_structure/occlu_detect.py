#! /usr/bin/python
# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: occlu_detect.py
Author: Yang Li
Date: 2018/11/10 17:43:31
Description: Occlusion Detection
"""

import cv2
from keras.models import load_model
from keras import backend as K
import os

from config.init_param import occlu_param, data_param
from model_structure.base_model import Model
from ml import metric_compute
from ml import classify
from utils import binary
from utils import logger
from utils import set_gpu


class OcclusionDetection(Model, object):
    def __init__(self):
        train_dir = os.path.join(data_param['data_save_dir'], 'train')
        super(OcclusionDetection, self).__init__(
            lr=occlu_param['init_lr'],
            epochs=occlu_param['epochs'],
            bs=occlu_param['bs'],
            model_name=occlu_param['model_name'],
            loss=occlu_param['loss'],
            metrics=["accuracy"],
            steps_per_epochs=len(os.listdir(train_dir)) // (occlu_param['bs'] * 6),
            classes=data_param['landmark_num'],
            final_act="sigmoid",
        )

    @staticmethod
    def val_compute(val_load, ext_lists, label_ext, gpu_ratio=0.5):
        # set gpu usage
        set_gpu(ratio=gpu_ratio)

        model = load_model(
            os.path.join(data_param['model_dir'], occlu_param['model_name']))
        val_data, val_labels = val_load(data_dir=os.path.join(data_param['data_save_dir'], 'val'),
                                        ext_lists=ext_lists,
                                        label_ext=label_ext,
                                        print_debug=data_param['print_debug'])

        # forward
        predict_labels = []
        length = len(val_data)
        for index in range(length):
            prediction = [binary(_, threshold=0.5)
                          for _ in classify(model, val_data[index])]
            predict_labels.append(prediction)
            if data_param['print_debug'] and (index + 1) % 500 == 0:
                logger("predicted {} imgs".format(index + 1))
        K.clear_session()

        # compute
        metric_compute(val_labels, predict_labels)
