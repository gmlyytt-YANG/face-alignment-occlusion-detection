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
import numpy as np
from keras import backend as K
from utils import logger

from config.init_param import data_param


def landmark_loss(y_true, y_pred):
    """Self defined loss function of landmark"""
    landmark_true = K.reshape(y_true, (-1, data_param['landmark_num'], 2))
    landmark_pred = K.reshape(y_pred, (-1, data_param['landmark_num'], 2))
    left_eye = K.mean(landmark_true[:, 36:42, :], axis=1)
    right_eye = K.mean(landmark_true[:, 42:48, :], axis=1)
    loss = K.mean(K.mean(K.sqrt(K.sum((landmark_true - landmark_pred) ** 2, axis=-1)), axis=-1) / K.sqrt(
        K.sum((right_eye - left_eye) ** 2)), axis=-1)
    return loss


def landmark_loss_compute(prediction, label):
    """loss compute of landmark_loss"""
    landmark_true = np.reshape(label, (data_param['landmark_num'], 2))
    landmark_pred = np.reshape(prediction, (data_param['landmark_num'], 2))
    left_eye = np.mean(landmark_true[36:42, :], axis=0)
    right_eye = np.mean(landmark_true[42:48, :], axis=0)
    loss = np.mean(np.sqrt(np.sum((landmark_true - landmark_pred) ** 2, axis=-1)), axis=-1) / np.sqrt(
        np.sum((right_eye - left_eye) ** 2))
    return loss


def scale(data):
    """Scale data to [0, 255] using min max method. """
    return np.multiply((data - np.min(data)) / (np.max(data) - np.min(data)),
                       255).astype(int)


def gaussian_noise(img, mode='gaussian', color=False):
    """Add gaussian noise to images """
    noised_img = None
    if mode == "gaussian":
        if color:
            row, col, channel = img.shape
        else:
            row, col = img.shape
        mean = 0
        var = 5
        sigma = var ** 0.5
        if color:
            gauss = np.random.normal(mean, sigma, (row, col, channel))
            gauss = gauss.reshape(row, col, channel)
        else:
            gauss = np.random.normal(mean, sigma, (row, col))
            gauss = gauss.reshape(row, col)
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


def metric_compute(val_labels, predict_labels):
    logger("the result of prediction of validation is as follow:")
    occlu_ratio = occlu_ratio_compute(val_labels)
    accuracy = accuracy_compute(val_labels, predict_labels)
    occlu_recall = recall_compute(val_labels, predict_labels, mode="occlu")
    clear_recall = recall_compute(val_labels, predict_labels, mode="clear")
    print("occlu_ratio is {}".format(occlu_ratio))
    print("accuracy is {}".format(accuracy))
    print("occlu_recall is {}".format(occlu_recall))
    print("clear_recall is {}".format(clear_recall))


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
