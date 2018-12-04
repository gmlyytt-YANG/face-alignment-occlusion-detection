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
import pickle

from config.init_param import occlu_param, data_param
from model_structure.base_model import Model
from ml import accuracy_compute
from ml import occlu_ratio_compute
from ml import recall_compute
from utils import binary
from utils import logger
from utils import set_gpu
from utils import heat_map_compute


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

    def val_compute(self, val_load, ext_lists, label_ext, normalizer=None, gpu_ratio=0.5):
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
                          for _ in self.classify(model, val_data[index])]
            predict_labels.append(prediction)
            if data_param['print_debug'] and (index + 1) % 500 == 0:
                logger("predicted {} imgs".format(index + 1))
        K.clear_session()

        # compute
        logger("the result of prediction of validation is as follow:")
        occlu_ratio = occlu_ratio_compute(val_labels)
        accuracy = accuracy_compute(val_labels, predict_labels)
        occlu_recall = recall_compute(val_labels, predict_labels, mode="occlu")
        clear_recall = recall_compute(val_labels, predict_labels, mode="clear")
        print("occlu_ratio is {}".format(occlu_ratio))
        print("accuracy is {}".format(accuracy))
        print("occlu_recall is {}".format(occlu_recall))
        print("clear_recall is {}".format(clear_recall))

    def test(self, model, img, landmark, is_heat_map=False, binary_output=False, normalize=False):
        img = cv2.resize(img, (data_param['img_size'], data_param['img_size']))
        if normalize:
            f_mean = open(os.path.join(data_param['normalizer_dir'], "normalizer.pkl"), 'rb')
            normalizer = pickle.load(f_mean)
            img = normalizer.transform(img)
        net_input = img
        if is_heat_map:
            net_input = heat_map_compute(img, landmark,
                                         landmark_is_01=False,
                                         img_color=True,
                                         radius=occlu_param['radius'])
        if binary_output:
            return [binary(_, threshold=0.5) for _ in self.classify(model, net_input)]
        return self.classify(model, net_input)
