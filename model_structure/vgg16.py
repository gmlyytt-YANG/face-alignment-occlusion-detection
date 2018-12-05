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

from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
import keras.backend as K
import tensorflow as tf

from config.init_param import occlu_param, data_param


class Vgg16Base(object):
    @staticmethod
    def build(width, height, depth, classes, weights='imagenet'):
        input_shape = (height, width, depth)
        if weights not in {'imagenet', None}:
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization) or `imagenet` '
                             '(pre-training on ImageNet).')

        model = Sequential()

        # Block 1
        model.add(Conv2D(64, (3, 3), input_shape=input_shape,
                         activation='relu', padding='same', name='block1_conv1'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

        # Block 2
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

        # Block 3
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

        # Block 4
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

        # Block 5
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

        return model

    @staticmethod
    def load_weights(model):
        if not os.path.exists(occlu_param['weight_path']):
            raise ValueError("there is no vgg16 weights data!")

        model.load_weights(os.path.join(occlu_param['weight_path'], occlu_param['weight_name']), by_name=True)

        return model


class Vgg16CutFC2(Vgg16Base, object):
    def build(self, width, height, depth, classes, final_act="softmax", mean_shape=None, weights='imagenet'):
        model = super(Vgg16CutFC2, self).build(
            width=width,
            height=height,
            depth=depth,
            classes=classes,
            weights=weights
        )

        # Classification block
        model.add(Flatten(name='flatten'))
        model.add(Dense(4096, activation='relu', name='fc1_self'))
        # model.add(Dense(4096, activation='relu', name='fc2'))
        model.add(Dense(classes, activation=final_act, name='predictions_self'))

        return self.load_weights(model)


class Vgg16Regress(Vgg16Base, object):
    def build(self, width, height, depth, classes, final_act=None, mean_shape=None, weights='imagenet'):
        model = super(Vgg16Regress, self).build(
            width=width,
            height=height,
            depth=depth,
            classes=classes,
            weights=weights
        )

        if mean_shape is not None:
            mean_shape = K.reshape(mean_shape, [classes, 1])
            mean_shape_tensor = K.variable(value=mean_shape)

        # Regression block
        model = Flatten(name='flatten')(model)
        model = Dense(4096, activation='relu', name='fc1_self')(model)
        # model.add(Dense(4096, activation='relu', name='fc2_self'))
        model = Dense(classes, name='predictions_self')(model)
        model = K.identity(model+mean_shape_tensor, name="landmark")

        return self.load_weights(model)
