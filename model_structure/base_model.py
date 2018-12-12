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

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import os

from config.init_param import data_param
from utils import logger

matplotlib.use('agg')


class Model(object):
    """Base Model"""

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

    def train(self, model_structure, train_load, val_load, img_ext_lists,
              train_data_dir, val_data_dir, label_ext, flatten=False,
              normalizer=None, weight_path=None):
        """Train procedure"""
        # load data
        logger('loading data')
        print(self.lr, self.epochs, self.bs, self.model_name, self.loss, self.steps_per_epoch)
        val_data, val_labels = val_load(data_dir=val_data_dir, img_ext_lists=img_ext_lists,
                                        label_ext=label_ext, flatten=flatten,
                                        normalizer=normalizer, print_debug=data_param['print_debug'])

        # build model
        logger('building model')
        opt = Adam(lr=self.lr, decay=self.lr / self.epochs)
        model = model_structure.build(width=data_param['img_size'], height=data_param['img_size'],
                                      depth=data_param['channel'], classes=self.classes,
                                      final_act=self.final_act, weight_path=weight_path)
        model.compile(loss=self.loss, optimizer=opt, metrics=self.metrics)

        logger('training')
        if not os.path.exists(data_param['model_dir']):
            os.makedirs(data_param['model_dir'])
        checkpoint = \
            ModelCheckpoint(filepath=os.path.join(data_param['model_dir'], self.model_name), save_best_only=True)
        # early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=2)
        # callback_list = [checkpoint, early_stopping]
        callback_list = [checkpoint]
        H = model.fit_generator(
            train_load(batch_size=self.bs, data_dir=train_data_dir,
                       img_ext_lists=img_ext_lists, label_ext=label_ext, flatten=flatten),
            validation_data=(val_data, val_labels),
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs, verbose=1, callbacks=callback_list)

        x_list = np.arange(0, len(H.history['loss']))
        y_1_list = H.history['loss']
        y_2_list = H.history['val_loss']
        plt.plot(x_list, y_1_list, 'g*-', label='train_loss')
        plt.plot(x_list, y_2_list, 'r*-', label='val_loss')
		# plt.plot(np.arange(0, len(H.history['acc'])), H.history['acc'], label='train_acc')
        # plt.plot(np.arange(0, len(H.history['val_acc'])), H.history['val_acc'], label='val_acc')
        plt.title('Training Loss and Accuracy')
        plt.xlabel('Epoch #')
        plt.ylabel('Loss/Accuracy')
        plt.legend(loc='upper right')
        filename = os.path.splitext(os.path.join(data_param['record_dir'], self.model_name))[0]
        np.savetxt(filename, H.history['val_loss'], delimiter=',')
        print('min val loss of {} is {}'.format(filename + '.txt', np.min(y_2_list)))
        plt.savefig('{}'.format(filename + '.png'))
