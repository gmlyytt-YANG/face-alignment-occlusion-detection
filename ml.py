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
Date: 2018/11/28 15:20:31
Description: Machine Learning Method
"""
from skimage.util import random_noise

from utils import *


def scale(data):
    return np.multiply((data - np.min(data)) / (np.max(data) - np.min(data)),
                       255).astype(int)


def gaussian_noise(img, mode='gaussian'):
    """Add gaussian noise to images """
    noised_img = None
    if mode == "gaussian":
        row, col, channel = img.shape
        mean = 0
        var = 5
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, channel))
        gauss = gauss.reshape(row, col, channel)
        noised_img = img + gauss
    noised_img = scale(noised_img)
    # show(noised_img)
    return noised_img


def occlu_ratio_compute(labels):
    count = 0
    for index in range(len(labels)):
        if np.sum(labels[index]) > 0:
            count += 1
    return float(count) / len(labels)


def accuracy_compute(labels, predictions):
    count = 0
    for index in range(len(labels)):
        if (labels[index] == predictions[index]).all():
            count += 1
    result = float(count) / len(labels)
    return result


def recall_compute(labels, predictions, mode="occlu"):
    count = 0
    good_count = 0
    if mode == "occlu":
        flag = 2
    else:
        flag = 0
    for index in range(len(labels)):
        for index_inner in range(len(labels[index])):
            if labels[index][index_inner] + \
                    predictions[index][index_inner] == flag:
                good_count += 1
            if labels[index][index_inner] == int(flag / 2):
                count += 1
    return float(good_count) / count


class StdMinMaxScaler(object):
    """Whiten and scale to [0, 255] dataset"""

    def __init__(self):
        self.mean_data = None
        self.std_data = None

    @staticmethod
    def _minmax(data):
        for index in range(len(data)):
            data[index] = scale(data[index])
        return data

    def fit_transform(self, data):
        data = self.fit(data)
        data = self._minmax(data)

        return data

    def fit(self, data):
        self.mean_data = np.mean(data, axis=0)
        data = data - self.mean_data

        self.std_data = np.std(data, axis=0)
        data = data / self.std_data

        return data

    def transform(self, data):
        data = data - self.mean_data
        data = data / self.std_data
        data = self._minmax(data)

        return data
