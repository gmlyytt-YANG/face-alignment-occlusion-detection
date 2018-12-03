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
from utils import *


# def load_landmark(label_name, dtype):

def load_occlu(label_name, dtype):
    return np.genfromtxt(label_name, dtype=dtype)


def load_img_label(img_name_list, label_name_list, load_label,
                   chosen_indices, print_debug=False):
    count = 0
    img_list = []
    label_list = []
    for index in chosen_indices:
        img = cv2.imread(img_name_list[index])
        occlusion = load_label(label_name_list[index], dtype=int)
        img_list.append(img)
        label_list.append(occlusion)
        if print_debug and (count + 1) % 500 == 0:
            logger("loaded {} data".format(count + 1))
        count += 1
    return img_list, label_list


def train_data_feed(batch_size, data_dir, ext_lists, label_ext):
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
        if label_ext == ".opts":
            load_label = load_occlu
        elif label_ext == ".pts":
            load_label = load_landmark
        img_list, occlusion_list = \
            load_img_label(img_name_list, label_name_list, load_label,
                           chosen_indices, print_debug=False)
        yield np.array(img_list), np.array(occlusion_list)


def validation_data_feed(data_dir, ext_lists, label_ext, print_debug=False):
    img_name_list, label_name_list = \
        get_filenames(data_dir, ext_lists, label_ext)

    data_size = len(img_name_list)
    img_list, label_list = \
        load_img_label(img_name_list, label_name_list, load_occlu,
                       range(data_size), print_debug=print_debug)

    return np.array(img_list), np.array(label_list)
