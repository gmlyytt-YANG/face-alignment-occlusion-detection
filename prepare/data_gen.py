#! /usr/bin/python
# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: data_gen.py
Author: Yang Li
Date: 2018/11/20 14:58:31
Description: Data Generator
"""
import cv2
import numpy as np

from utils import logger
from utils import get_filenames


def load_landmark(label_name, mean_shape=None):
    landmarks = np.genfromtxt(label_name)
    if mean_shape:
        return landmarks.flatten() - mean_shape.flatten()
    return landmarks.flatten()


def load_occlu(label_name, mean_shape=None):
    print(label_name)
    return np.genfromtxt(label_name)


def load_img_label(img_name_list, label_name_list, load_label,
                   chosen_indices, mean_shape=None, normalizer=None, print_debug=False):
    """Load faces and labels

    :param: img_name_list: face or heatmap img path name list.
    :param: label_name_list: occlu or landmark path name list.
    :param: load_label: load label function.
    :param: chosen_indices:
    :param: mean_shape:
    :param: normalizer:
    :param: print_debug:
    """
    count = 0
    img_list = []
    label_list = []
    for index in chosen_indices:
        img = cv2.imread(img_name_list[index])
        if normalizer:
            img = normalizer.transform(img)
        label = load_label(label_name_list[index], mean_shape)
        img_list.append(img)
        label_list.append(label)
        if print_debug and (count + 1) % 500 == 0:
            logger("loaded {} data".format(count + 1))
        count += 1
        # if (index + 1) > 1000:
        #     break
    return np.array(img_list), np.array(label_list)


def train_data_feed(batch_size, data_dir, ext_lists, label_ext, mean_shape=None):
    """Train data feed.

    :param: batch_size:
    :param: data_dir:
    :param: ext_lists: img suffix lists.
    :param: label_ext: label suffix.
    :param: mean_shape:
    :param: print_debug:
    """
    img_name_list, label_name_list = \
        get_filenames(data_dir, ext_lists, label_ext)
    data_size = len(img_name_list)
    batch_offset = 0
    indices = [_ for _ in range(data_size)]
    while True:
        start = batch_offset
        batch_offset += batch_size
        if batch_offset > data_size:
            np.random.shuffle(indices)
            start = 0
            batch_offset = batch_size
        end = batch_offset
        chosen_indices = indices[start: end]
        load_label = load_occlu
        if label_ext == ".pts":
            load_label = load_landmark
        img_list, label_list = load_img_label(img_name_list=img_name_list,
                                              label_name_list=img_name_list,
                                              load_label=load_label,
                                              chosen_indices=chosen_indices,
                                              mean_shape=mean_shape,
                                              print_debug=False)
        yield img_list, label_list


def val_data_feed(data_dir, ext_lists, label_ext,
                  mean_shape=None, normalizer=None, print_debug=False):
    """Validation data feed

    :param: data_dir:
    :param: ext_lists: img suffix lists.
    :param: label_ext: label suffix.
    :param: mean_shape:
    :param: normalizer:
    :param: print_debug:
    """
    img_name_list, label_name_list = \
        get_filenames(data_dir, ext_lists, label_ext)

    data_size = len(img_name_list)
    load_label = load_occlu
    if label_ext == ".pts":
        load_label = load_landmark
    img_list, label_list = load_img_label(img_name_list=img_name_list,
                                          label_name_list=label_name_list,
                                          load_label=load_label,
                                          chosen_indices=range(data_size),
                                          mean_shape=mean_shape,
                                          normalizer=normalizer,
                                          print_debug=print_debug)

    return img_list, label_list
