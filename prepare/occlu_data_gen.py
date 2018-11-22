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

import numpy as np
import os
import pickle

from prepare.utils import logger


def train_data_feed(batch_size, data_dir):
    data_size = len(os.listdir(data_dir))
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
        # print("\n")
        # logger(format(" ".join([str(_) for _ in chosen_indices])))
        # logger("chosed indices are {}".format(" ".join(chosen_indices)))
        data = []
        labels = []
        for index in chosen_indices:
            f_dataset = open(os.path.join(data_dir, "{}.pkl".format(index)), 'rb')
            base = pickle.load(f_dataset)
            data.append(np.multiply(base['image'], 255).astype(int))
            labels.append(base['label'])
            f_dataset.close()
        yield np.array(data), np.array(labels)


def validation_data_feed(data_dir, print_debug=False):
    data_list = []
    labels_list = []
    count = 0
    for path in os.listdir(data_dir):
        data_path = os.path.join(data_dir, path)
        f_data = open(data_path, 'rb')
        data = pickle.load(f_data)
        data_list.append(np.multiply(data['image'], 255).astype(int))
        labels_list.append(data['label'])
        f_data.close()
        if print_debug:
            if (count + 1) % 500 == 0:
                logger("loaded {} data in phase validation".format(count + 1))
        count = count + 1
        # if count > 1000:
        #     break
    return np.array(data_list), np.array(labels_list)
