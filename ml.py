#! /usr/bin/python
# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: ml.py
Author: Yang Li
Date: 2018/11/14 15:04:31
Description: machine learning related functions
"""

import numpy as np
from sklearn.model_selection import train_test_split
from utils import logger


def dataset_split(x, y, test_size=0.3, random_state=0):
    """Dataset Splitting using sklearn

    :param x: image dataset
    :param y: label
    :param test_size:
    :param random_state:
    :return:
    """
    assert len(x) == len(y)
    dataset_size = len(x)
    x_reshaped = []

    # row-trans dataset
    img_size = [0, 0]
    for index in range(dataset_size):
        if index == 0:
            img_size = x[index].shape
        else:
            if x[index].shape != img_size:
                logger("matrix dim in img_size")
                continue
        one_instance = x[index]
        x_reshaped.append(np.reshape(one_instance, (1, img_size[0] * img_size[1])))
        y[index] = int(y[index])

    # data splitting
    x_train, x_test, y_train, y_test = train_test_split(x_reshaped, y,
                                                        shuffle=True,
                                                        test_size=test_size,
                                                        random_state=random_state)
    data_container = {
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test,
    }
    return data_container
