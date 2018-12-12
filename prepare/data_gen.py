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


def load_label(label_name, flatten=False):
    landmarks = np.genfromtxt(label_name)
    if flatten:
        return landmarks.flatten()
    return landmarks


def load_img_label(img_name_list, label_name_list,
                   chosen_indices, flatten=False, normalizer=None, print_debug=False):
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
        label = load_label(label_name_list[index], flatten)
        img_list.append(img)
        label_list.append(label)
        if print_debug and (count + 1) % 500 == 0:
            logger("loaded {} data".format(count + 1))
        count += 1
        # if (index + 1) > 1000:
        #     break
    return np.array(img_list), np.array(label_list)


def train_data_feed(batch_size, data_dict=None):
    """Train data feed.

    :param: batch_size:
    :param: data_dir:
    :param: img_ext_lists: img suffix lists.
    :param: label_ext: label suffix.
    :param: mean_shape:
    :param: print_debug:
    """
    data_dir = data_dict['data_dir']
    img_ext_lists = data_dict['img_ext_lists']
    label_ext = data_dict['label_ext']
    flatten = data_dict['flatten']
    img_name_list, label_name_list = \
        get_filenames([data_dir], img_ext_lists, label_ext)
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
        img_list, label_list = \
            load_img_label(img_name_list=img_name_list, label_name_list=label_name_list,
                           chosen_indices=chosen_indices, flatten=flatten, print_debug=False)
        # for index in range(len(label_list)):
        #     label_list[index] = [round(_, 2) for _ in label_list[index]]
        # label_list = np.array(label_list)

        yield img_list, label_list


def val_data_feed(data_dict=None):
    """Validation data feed

    :param: data_dir:
    :param: img_ext_lists: img suffix lists.
    :param: label_ext: label suffix.
    :param: mean_shape:
    :param: normalizer:
    :param: print_debug:
    """
    data_dir = data_dict['data_dir']
    img_ext_lists = data_dict['img_ext_lists']
    label_ext = data_dict['label_lists']
    flatten = data_dict['flatten']
    normalizer = data_dict['normalizer']
    print_debug = data_dict['print_debug']
    img_name_list, label_name_list = \
        get_filenames([data_dir], img_ext_lists, label_ext)

    data_size = len(img_name_list)
    img_list, label_list = \
        load_img_label(img_name_list=img_name_list, label_name_list=label_name_list,
                       chosen_indices=range(data_size), flatten=flatten,
                       normalizer=normalizer, print_debug=print_debug)

    return img_list, label_list
