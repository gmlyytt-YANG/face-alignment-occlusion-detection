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
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.models import Model


class Vgg16Base(object):
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height, width, depth)
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
    def load_weights(model, weight_path):
        if not os.path.exists(weight_path):
            raise ValueError("there is no vgg16 weights data!")

        model.load_weights(weight_path, by_name=True)

        return model


class Vgg16CutFC2(Vgg16Base, object):
    def build(self, width, height, depth, classes, final_act="softmax", weight_path=None):
        x, input = super(Vgg16CutFC2, self).build(
            width=width,
            height=height,
            depth=depth,
            classes=classes,
        )

        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1_self')(x)
        # model.add(Dense(4096, activation='relu', name='fc2'))
        x = Dense(classes, activation=final_act, name='predictions_self')(x)

        model = Model(input, x)

        return self.load_weights(model, weight_path=weight_path)


class Vgg16Regress(Vgg16Base, object):
    def build(self, width, height, depth, classes, final_act=None, weight_path=None):
        x, input = super(Vgg16Regress, self).build(
            width=width,
            height=height,
            depth=depth,
            classes=classes,
        )

        # Regression block
        x = Flatten(name='flatten')(x)
        x = Dense(512, name='fc1_self')(x)
        # x = Dense(1000, activation='relu', name='fc1_self')(x)
        x = Dense(classes, name='predictions_self')(x)
        model = Model(input, x)

        return self.load_weights(model, weight_path=weight_path)
