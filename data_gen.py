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
import os

from config.init_param import data_param, occlu_param
from utils import get_filenames
from utils import normalize_data
from utils import load_basic_info
from utils import get_face
from utils import heat_map_compute
from utils import logger


def load_label(label_name, flatten=False):
    landmarks = np.genfromtxt(label_name)
    if flatten:
        return landmarks.flatten()
    return landmarks


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
        img_list = []
        label_list = []
        for index in chosen_indices:
            img = cv2.imread(img_name_list[index])
            label = load_label(label_name_list[index], flatten)
            img_list.append(img)
            label_list.append(label)

        yield np.array(img_list), np.array(label_list)


def val_data_feed(data_dict=None):
    data_dir = data_dict['data_dir']
    img_ext_lists = data_dict['img_ext_lists']
    label_ext = data_dict['label_ext']
    flatten = data_dict['flatten']
    img_name_list, label_name_list = \
        get_filenames([data_dir], img_ext_lists, label_ext)
    data_size = len(img_name_list)
    img_list = []
    label_list = []
    for index in range(data_size):
        img = cv2.imread(img_name_list[index])
        label = load_label(label_name_list[index], flatten)
        img_list.append(img)
        label_list.append(label)
    return np.array(img_list), np.array(label_list)


def wdpts_process(label_path):
    landmark = np.genfromtxt(label_path)
    return landmark


def opts_process(label_path, bbox, img_size):
    landmark_ori = np.genfromtxt(label_path)
    landmark = normalize_data(landmark_ori, bbox, occlu_include=True)
    landmark = np.multiply(np.clip(landmark[:, :2], 0, 1), img_size)
    occlu = landmark_ori[:, -1]
    return landmark, occlu


def pts_process(label_path, bbox, img_size):
    landmark_ori = np.genfromtxt(label_path, skip_header=3, skip_footer=1)
    landmark = np.multiply(np.clip(normalize_data(landmark_ori, bbox, occlu_include=False, label_ext=".pts"), 0, 1),
                           img_size)
    return landmark


def load_imgs_labels_core(img_path, bbox, img_size, normalizer=None, label_ext=".pts"):
    img = cv2.imread(img_path)
    face = cv2.resize(get_face(img, bbox, need_to_convert_to_int=True),
                      (img_size, img_size))
    if normalizer is not None:
        face = normalizer.transform(face)

    label_path = os.path.splitext(img_path)[0] + label_ext
    if label_ext == '.wdpts':
        landmark = wdpts_process(label_path)
    elif label_ext == '.opts':
        landmark, occlu = opts_process(label_path, bbox, img_size)
        # print(landmark)
        # print(occlu)
        # print('---------')
        return face, landmark, occlu
    elif label_ext == '.pts':
        landmark = pts_process(label_path, bbox, img_size)
    else:
        raise ValueError('there is no such exts')
    return face, landmark


def load_imgs_labels(img_root=None, img_size=None, occlu_include=False, flatten=False,
                     normalizer=None, chosen="random", label_ext=".pts", data_dict=None):
    """Load imgs and labels based on mat file

    :param img_root: img root dir
    :param img_size:
    :param normalizer:
    :param occlu_include:
    :param chosen: whether to choose specific indices of dataset or just random
    :param label_ext:
    :param data_dict:

    :return chosen objs
    """
    if data_dict is not None:
        img_root = data_dict['img_root']
        img_size = data_dict['img_size']
        occlu_include = data_dict['occlu_include']
        normalizer = data_dict['normalizer']
        chosen = data_dict['chosen']
        label_ext = data_dict['label_ext']
        flatten = data_dict['flatten']
    # print(label_ext)
    img_paths, bboxes = load_basic_info('raw_300W_release.mat', img_root)
    if chosen == "random":
        length = len(img_paths)
        index = np.random.randint(0, length)
        return load_imgs_labels_core(img_path=img_paths[index], bbox=bboxes[index],
                                     img_size=img_size, normalizer=normalizer, label_ext=label_ext)
    else:
        faces = []
        labels = []
        occlus = []
        for index in chosen:
            _ = load_imgs_labels_core(img_path=img_paths[index], bbox=bboxes[index],
                                      img_size=img_size, normalizer=normalizer, label_ext=label_ext)
            # show(face)
            faces.append(_[0])
            label = _[1]
            if flatten:
                label = label.flatten()
            labels.append(label)
            if occlu_include:
                occlus.append(_[2])
        if occlu_include:
            return np.array(faces), np.array(labels), np.array(occlus)
        else:
            return np.array(faces), np.array(labels)


def load_imgs_occlus(data_dict=None):
    if not data_dict['occlu_include']:
        raise ValueError('invalid input!')
    faces, landmarks, occlus = load_imgs_labels(data_dict=data_dict)
    # print(len(faces), len(landmarks), len(occlus))
    if data_dict['is_heatmap']:
        heatmaps = []
        for (face, landmark) in zip(faces, landmarks):
            heatmap = heat_map_compute(face, landmark, landmark_is_01=False,
                                       img_color=True, radius=occlu_param['radius'])
            heatmaps.append(heatmap)
            if data_param['print_debug'] and len(heatmaps) % 100 == 0:
                logger('processed {} heatmaps'.format(len(heatmaps)))
            # if len(heatmaps) > 20:
            #     break
        return np.array(heatmaps), occlus
    return faces, occlus
