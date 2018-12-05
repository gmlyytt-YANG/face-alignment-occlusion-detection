#! /usr/bin/python
# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: vgg16.py
Author: Yang Li
Date: 2018/11/26 10:22:31
Description: self-define VGG16NET
"""

import os
import numpy as np
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Lambda
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.models import Model
import keras.backend as K

from config.init_param import occlu_param, data_param


class Vgg16Base(object):
    @staticmethod
    def build(width, height, depth, classes, weights='imagenet'):
        input_shape = (height, width, depth)
        if weights not in {'imagenet', None}:
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization) or `imagenet` '
                             '(pre-training on ImageNet).')

        input = Input(input_shape)
        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        return x, input

    @staticmethod
    def load_weights(model):
        if not os.path.exists(occlu_param['weight_path']):
            raise ValueError("there is no vgg16 weights data!")

        model.load_weights(os.path.join(occlu_param['weight_path'], occlu_param['weight_name']), by_name=True)

        return model


class Vgg16CutFC2(Vgg16Base, object):
    def build(self, width, height, depth, classes, final_act="softmax", mean_shape=None, weights='imagenet'):
        x, input = super(Vgg16CutFC2, self).build(
            width=width,
            height=height,
            depth=depth,
            classes=classes,
            weights=weights
        )

        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1_self')(x)
        # model.add(Dense(4096, activation='relu', name='fc2'))
        x = Dense(classes, activation=final_act, name='predictions_self')(x)

        model = Model(input, x)

        return self.load_weights(model)


class Vgg16Regress(Vgg16Base, object):
    def build(self, width, height, depth, classes, final_act=None, mean_shape=None, weights='imagenet'):
        x, input = super(Vgg16Regress, self).build(
            width=width,
            height=height,
            depth=depth,
            classes=classes,
            weights=weights
        )

        if mean_shape is not None:
            mean_shape = K.constant(np.reshape(mean_shape, [1, 2]))
            mean_shape = K.reshape(mean_shape, [data_param['landmark_num'], 1])
            mean_shape_tensor = K.variable(value=mean_shape, dtype=K.floatx())

        # Regression block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1_self')(x)
        # model.add(Dense(4096, activation='relu', name='fc2_self'))
        x = Dense(classes, name='predictions_self')(x)
        x = K.identity(x + mean_shape_tensor, name="landmark")

        model = Model(input, x)

        return self.load_weights(model)
