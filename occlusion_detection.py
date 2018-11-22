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
Date: 2018/11/10 17:43:31
Description: Data Augment
"""

import cv2
import numpy as np
import os

from keras.models import load_model
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense

from config.init_param import occlu_param
from model_structure.occlu_model import SmallerVGGNet
from prepare.utils import logger, load_imgs
from prepare.occlu_data_gen import train_data_feed, validation_data_feed
from prepare.ImageServer import ImageServer


class OcclusionDetection(object):
    @staticmethod
    def data_pre():
        # load data(img, bbox, pts)
        mat_file = os.path.join(occlu_param['img_root_dir'], 'raw_300W_release.mat')
        img_paths, bboxes = load_imgs(mat_file)
        # data prepare
        img_server = ImageServer(data_size=len(img_paths),
                                 img_size=occlu_param['img_size'], color=True)
        img_server.process(img_root=occlu_param['img_root_dir'], img_paths=img_paths,
                           bounding_boxes=bboxes, print_debug=occlu_param['print_debug'])

        # splitting
        logger("train validation splitting")
        img_server.train_validation_split(test_size=occlu_param['test_size'],
                                          random_state=occlu_param['random_state'])

        # saving
        logger("saving data")
        img_server.save(occlu_param['data_save_dir'], print_debug=occlu_param['print_debug'])

    @staticmethod
    def train():
        # loading data
        logger("loading data")
        train_dir = os.path.join(occlu_param['data_save_dir'], "train")
        validation_dir = os.path.join(occlu_param['data_save_dir'], "validation")
        validation_data, validation_labels = validation_data_feed(validation_dir,
                                                                  print_debug=occlu_param['print_debug'])
        logger("the occlusion ratio is {}"
               .format(float(np.sum(validation_labels)) / (validation_labels.shape[0] * validation_labels.shape[1])))

        # build model
        logger("building model")
        model = SmallerVGGNet.build(
            width=occlu_param['img_size'], height=occlu_param['img_size'],
            depth=occlu_param['channel'], classes=occlu_param['landmark_num'],
            final_act="sigmoid")
        # base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        # predictions = Dense(68, activation='softmax')(base_model.output)
        # model = Model(inputs=base_model.input, outputs=predictions)
        opt = Adam(lr=occlu_param['init_lr'], decay=occlu_param['init_lr'] / occlu_param['epochs'])
        model.compile(loss="binary_crossentropy", optimizer=opt,
                      metrics=["accuracy"])

        # train model
        logger("training")
        H = model.fit_generator(
            train_data_feed(occlu_param['bs'], train_dir),
            validation_data=(validation_data, validation_labels),
            steps_per_epoch=len(os.listdir(train_dir)) // occlu_param['bs'],
            epochs=occlu_param['epochs'], verbose=1)

        # save model
        logger("saving model")
        if not os.path.exists(occlu_param['model_dir']):
            os.mkdir(occlu_param['model_dir'])
        model.save(os.path.join(occlu_param['model_dir'], occlu_param['model_name']))

    @staticmethod
    def classify(img, need_to_normalize=False):
        img = cv2.resize(img, (occlu_param['img_size'], occlu_param['img_size']))
        if need_to_normalize:
            img = img.astype("float") / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        model = load_model(os.path.join(occlu_param['model_dir'], occlu_param['model_name']))
        prob = model.predict(img)[0]
        for index, elem in enumerate(prob):
            print("{} : {}".format(index + 1, elem))
