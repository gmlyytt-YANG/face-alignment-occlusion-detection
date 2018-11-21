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
    start = 0
    end = len(os.listdir(data_dir))
    indices = [_ for _ in range(end)]
    loop_count = end // batch_size
    count = 0
    while True:
        if count >= loop_count:
            np.random.shuffle(indices)
            start = 0
            count = 0

        chosen_indices = indices[start * batch_size: (start + 1) * batch_size]
        start += batch_size
        data = []
        labels = []
        count += 1
        for index in chosen_indices:
            f_dataset = open(os.path.join(data_dir, "{}.pkl".format(index)), 'rb')
            base = pickle.load(f_dataset)
            data.append(base['image'])
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
        data_list.append(data['image'])
        labels_list.append(data['label'])
        f_data.close()
        if print_debug:
            if (count + 1) % 500 == 0:
                logger("loaded {} data in phase validation".format(count + 1))
        count = count + 1
        if count > 1000:
            break
    return np.array(data_list), np.array(labels_list)
