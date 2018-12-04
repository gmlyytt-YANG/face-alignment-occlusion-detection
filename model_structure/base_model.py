#! /usr/bin/python
# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: base_model.py
Author: Yang Li
Date: 2018/11/10 17:43:31
Description: Base Model
"""

import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
import os

from config.init_param import data_param
from utils import set_gpu
from utils import logger

plt.switch_backend('agg')

class Model(object):
    def __init__(self, lr, epochs, bs, model_name,
                 loss, metrics, steps_per_epochs, classes, final_act=None):
        self.lr = lr
        self.epochs = epochs
        self.bs = bs
        self.model_name = model_name
        self.loss = loss
        self.metrics = metrics
        self.steps_per_epoch = steps_per_epochs
        self.classes = classes
        self.final_act = final_act

    def train(self, model_structure, train_load, val_load,
              ext_lists, label_ext, mean_shape=None, normalizer=None, gpu_ratio=0.5):
        # set gpu usage
        set_gpu(ratio=gpu_ratio)

        # load data
        logger('loading data')
        val_data, val_labels = val_load(data_dir=os.path.join(data_param['data_save_dir'], 'val'),
                                        ext_lists=ext_lists,
                                        label_ext=label_ext,
                                        mean_shape=mean_shape,
                                        normalizer=normalizer,
                                        print_debug=data_param['print_debug'])

        # build model
        logger('building model')
        opt = Adam(lr=self.lr, decay=self.lr / self.epochs)
        model = model_structure.build(width=data_param['img_size'],
                                      height=data_param['img_size'],
                                      depth=data_param['channel'],
                                      classes=self.classes,
                                      final_act=self.final_act)
        model.compile(loss=self.loss, optimizer=opt, metrics=self.metrics)

        logger('training')
        if not os.path.exists(data_param['model_dir']):
            os.makedirs(data_param['model_dir'])
        checkpoint = \
            ModelCheckpoint(filepath=os.path.join(data_param['model_dir'], self.model_name))
        early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=2)
        callback_list = [checkpoint, early_stopping]
        H = model.fit_generator(
            train_load(batch_size=self.bs, data_dir=os.path.join(data_param['data_save_dir'], 'train'),
                       ext_lists=ext_lists, label_ext=label_ext, mean_shape=mean_shape),
            validation_data=(val_data, val_labels),
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs, verbose=1, callbacks=callback_list)

        plt.plot(np.arange(0, self.epochs), H.history['loss'], label='train_loss')
        plt.plot(np.arange(0, self.epochs), H.history['val_loss'], label='val_loss')
        plt.plot(np.arange(0, self.epochs), H.history['acc'], label='train_acc')
        plt.plot(np.arange(0, self.epochs), H.history['val_acc'], label='val_acc')
        plt.title('Training Loss and Accuracy')
        plt.xlabel('Epoch #')
        plt.ylabel('Loss/Accuracy')
        plt.legend(loc='upper right')
        plt.savefig('{}'.format(self.model_name))

        K.clear_session()

    @staticmethod
    def classify(model, img):
        if img.shape[:2] != [data_param['img_size'], data_param['img_size']]:
            img = cv2.resize(img, (data_param['img_size'], data_param['img_size']))
        img = np.expand_dims(img_to_array(img), axis=0)
        return model.predict(img)[0]
