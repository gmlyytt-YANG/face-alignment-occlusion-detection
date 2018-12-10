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
import os

from config.init_param import occlu_param, data_param
from model_structure.base_model import Model
from ml import metric_compute
from ml import classify
from utils import binary
from utils import logger
from utils import heat_map_compute
from utils import count_file


class OcclusionDetection(Model, object):
    """Occlusion Detection Model"""

    def __init__(self):
        train_dir = os.path.join(data_param['data_save_dir'], 'train')
        train_num = count_file([train_dir], ["_heatmap.png", "_heatmap.jpg"])
        super(OcclusionDetection, self).__init__(
            lr=occlu_param['init_lr'],
            epochs=occlu_param['epochs'],
            bs=occlu_param['bs'],
            model_name=occlu_param['model_name'],
            loss=occlu_param['loss'],
            metrics=["accuracy"],
            steps_per_epochs=train_num // occlu_param['bs'],
            classes=data_param['landmark_num'],
            final_act="sigmoid",
        )

    @staticmethod
    def val_compute(val_load, ext_lists, label_ext, model=None):
        """Compute loss of imgs of ext_lists named label_ext"""
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

        # compute
        metric_compute(val_labels, predict_labels)

    @staticmethod
    def test(img, landmark, is_heat_map=False, binary_output=False, model=None):
        """Classify img"""
        net_input = img
        if is_heat_map:
            net_input = heat_map_compute(img, landmark,
                                         landmark_is_01=False,
                                         img_color=True,
                                         radius=occlu_param['radius'])
        if binary_output:
            return [binary(_, threshold=0.5) for _ in classify(model, net_input)]
        return classify(model, net_input)
