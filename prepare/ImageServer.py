#! /usr/bin/python
# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: ImageServer.py
Author: Yang Li
Date: 2018/11/17 19:05:31
Description: Data Preparation
"""

import cv2
import numpy as np
import pickle
import os

from prepare.utils import read_pts, logger, data_aug


class ImageServer(object):
    """Image Source Server

    Attributes:
        landmarks: annotations
        imgs:
        bounding_boxes: bbox list
        data_size:
        color:
        img_size: default size is [512, 512]
    """

    def __init__(self, data_size, img_size=None, landmark_size=68, color=False):
        self.landmarks = []
        self.faces = []
        self.aug_landmarks = []
        self.occlusions = []
        self.img_paths = []
        self.imgs = []
        self.bounding_boxes = []
        self.data_size = data_size
        self.landmark_size = landmark_size
        self.color = color
        self.img_size = img_size if img_size is not None else [512, 512]

    def prepare_data(self, img_root, img_paths, bounding_boxes, print_debug=False):
        """Getting data
        :param print_debug:
        :param bounding_boxes:
        :param img_root:
        :param img_paths:
        :return:
        """
        self.bounding_boxes = bounding_boxes
        for index in range(self.data_size):
            img_path = img_paths[index]
            prefix = img_path.split('.')[0]
            pts_path = prefix + '.pts_occlu'
            pts_path = os.path.join(img_root, pts_path)
            img_path = os.path.join(img_root, img_path)
            self.landmarks.append(read_pts(pts_path))
            self.img_paths.append(img_path)
            if print_debug:
                if (index + 1) % 100 == 0:
                    logger("processed {} basic infos".format(index + 1))

    def load_imgs(self, print_debug):
        """load imgs"""
        for index in range(self.data_size):
            # load img
            if self.color:
                img = cv2.imread(self.img_paths[index])
            else:
                img = cv2.imread(self.img_paths[index], cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger("{} img read error".format(self.img_paths[index]))
                continue
            self.imgs.append(img)
            bbox = [int(_) for _ in self.bounding_boxes[index]]

            # normalize landmark
            landmark = self.landmarks[index]
            x_normalized = (landmark[:, 0] - bbox[0]) / (bbox[1] - bbox[0])
            y_normalized = (landmark[:, 1] - bbox[2]) / (bbox[3] - bbox[2])
            landmark_normalized = np.stack((x_normalized, y_normalized, landmark[:, 2]), axis=1)
            del x_normalized, y_normalized

            # data augment
            face_dups, landmark_dups, occlusion_dups = data_aug(img,
                                                                landmark_normalized,
                                                                bbox,
                                                                self.img_size,
                                                                self.color)
            self.faces.extend(face_dups)
            self.aug_landmarks.extend(landmark_dups)
            self.occlusions.extend(occlusion_dups)
            if print_debug:
                if (index + 1) % 100 == 0:
                    logger("processed {} images".format(index + 1))
            if index > 100:
                break

    def save(self, dataset_path, file_name=None):
        """Save data"""
        if file_name is None:
            file_name = "dataset_nimgs={0}_size={1}".format(len(self.faces), self.img_size)
            if self.color:
                file_name += "_color={0}".format(self.color)
            file_name += ".npz"
        arrays = {key: value for key, value in self.__dict__.items()
                  if not key.startswith('__') and not callable(key)}
        np.savez(os.path.join(dataset_path, file_name), **arrays)
