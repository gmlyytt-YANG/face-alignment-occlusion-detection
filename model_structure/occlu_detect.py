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
from config.init_param import occlu_param, data_param
from model_structure.base_model import Model
from ml import classify
from utils import binary
from utils import logger
from ml import metric_compute
from utils import heat_map_compute


class OcclusionDetection(Model, object):
    """Occlusion Detection Model"""

    def __init__(self, lr=None, epochs=None, bs=None, model_name=None, loss=None, esm=None, train_num=None):
        super(OcclusionDetection, self).__init__(
            lr=lr,
            epochs=epochs,
            bs=bs,
            model_name=model_name,
            loss=loss,
            metrics=["accuracy"],
            steps_per_epochs=train_num // bs,
            esm=esm,
            classes=data_param['landmark_num'],
            final_act="sigmoid",
        )

    @staticmethod
    def val_compute(data, occlus, model, is_heatmap=False):
        predictions = []
        if is_heatmap:
            for face, landmark in data:
                prediction = \
                    OcclusionDetection.test(img=face, landmark=landmark, is_heatmap=is_heatmap,
                                            binary_output=True, model=model)
                # print(prediction)
                predictions.append(prediction)
                if data_param['print_debug'] and len(predictions) % 100 == 0:
                    logger("predicted {} imgs".format(len(predictions)))
        else:
            for face in data:
                prediction = \
                    OcclusionDetection.test(img=face, is_heatmap=is_heatmap,
                                            binary_output=True, model=model)
                # print(prediction)
                predictions.append(prediction)
                if data_param['print_debug'] and len(predictions) % 100 == 0:
                    logger("predicted {} imgs".format(len(predictions)))
        metric_compute(occlus, predictions)

    @staticmethod
    def test(img, landmark=None, is_heatmap=False, binary_output=False, model=None):
        """Classify img"""
        net_input = img
        if is_heatmap:
            net_input = heat_map_compute(img, landmark,
                                         landmark_is_01=False,
                                         img_color=True,
                                         radius=occlu_param['radius'])
        if binary_output:
            return [binary(_, threshold=0.5) for _ in classify(model, net_input)]
        return classify(model, net_input)
