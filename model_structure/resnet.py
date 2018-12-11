#! /usr/bin/python
# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: resnet.py
Author: Yang Li
Date: 2018/12/04 21:35:31
Description: self-define ResNet
"""

from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Add
from keras.layers import Input
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.models import Model


class ResNet(object):
    @staticmethod
    def load_weights(model, weight_path):
        if not os.path.exists(weight_path):
            raise ValueError("there is no vgg16 weights data!")

        model.load_weights(weight_path, by_name=True)

        return model

    @staticmethod
    def identity_block(model, f, filters, stage, block):
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2, F3 = filters

        # Save the input value. You'll need this later to add back to the main path.
        model_shortcut = model

        # First component of main path
        model.add(Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a'))
        model.add(BatchNormalization(axis=3, name=bn_name_base + '2a'))
        model.add(Activation('relu'))

        # Second component of main path (≈3 lines)
        model.add(Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b'))
        model.add(BatchNormalization(axis=3, name=bn_name_base + '2b'))
        model.add(Activation('relu'))

        # Third component of main path (≈2 lines)
        model.add(Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c'))
        model.add(BatchNormalization(axis=3, name=bn_name_base + '2c'))

        # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        model = Add()([model, model_shortcut])
        model.add(Activation('relu'))
        return model

    @staticmethod
    def convolutional_block(model, f, filters, stage, block, s=2):
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2, F3 = filters

        # Save the input value
        model_shortcut = model

        # MAIN PATH
        # First component of main path
        model.add(Conv2D(F1, (1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a'))
        model.add(BatchNormalization(axis=3, name=bn_name_base + '2a'))
        model.add(Activation('relu'))

        # Second component of main path (≈3 lines)
        model.add(Conv2D(F2, (f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b'))
        model.add(BatchNormalization(axis=3, name=bn_name_base + '2b'))
        model.add(Activation('relu'))

        # Third component of main path (≈2 lines)
        model.add(Conv2D(F3, (1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c'))
        model.add(BatchNormalization(axis=3, name=bn_name_base + '2c'))

        # SHORTCUT PATH (≈2 lines)
        model_shortcut.add(Conv2D(F3, (1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1'))
        model_shortcut.add(BatchNormalization(axis=3, name=bn_name_base + '1'))

        # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        model = Add()([model, model_shortcut])
        model.add(Activation('relu'))
        return model

    def build(self, width, height, depth, classes, final_act=None, weight_path=None): 
        input_shape = (height, width, depth)
        # Define the input as a tensor with shape input_shape
        X_input = Input(input_shape)

        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X_input)

        # Stage 1
        X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(X)
        X = BatchNormalization(axis=3, name='bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        # Stage 2
        X = self.convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
        X = self.identity_block(X, 3, [64, 64, 256], stage=2, block='b')
        X = self.identity_block(X, 3, [64, 64, 256], stage=2, block='c')

        # Stage 3 (≈4 lines)
        X = self.convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
        X = self.identity_block(X, 3, [128, 128, 512], stage=3, block='b')
        X = self.identity_block(X, 3, [128, 128, 512], stage=3, block='c')
        X = self.identity_block(X, 3, [128, 128, 512], stage=3, block='d')

        # Stage 4 (≈6 lines)
        X = self.convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

        # Stage 5 (≈3 lines)
        X = self.convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
        X = self.identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
        X = self.identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

        # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
        X = AveragePooling2D((2, 2), strides=(2, 2))(X)

        # output layer
        X = Flatten()(X)
        X = Dense(units=classes, name='prediction_self')(X)
        model = Model(inputs=X_input, outputs=X)

        return self.load_weights(model, weight_path=weight_path)
