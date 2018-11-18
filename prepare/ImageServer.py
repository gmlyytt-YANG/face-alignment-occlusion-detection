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
import os
from multiprocessing import Pool
import time

from prepare.utils import read_pts, logger, \
    data_aug, dataset_split, heat_map_compute, show


def heat_map_dist(point, matrix):
    sigma = 0.05
    D = np.min(np.sqrt(np.sum((point - matrix) ** 2, axis=1)))
    M = np.exp(np.multiply(D ** 2, - 2 * sigma ** 2))
    return M


class ImageServer(object):
    """Image Source Server

    Attributes:
        landmarks: annotations
        bounding_boxes: bbox list
        data_size:
        color:
        img_size: default size is [512, 512]
    """

    def __init__(self, data_size, img_size=None, landmark_size=68, color=False):
        self.landmarks = []
        self.faces = []
        self.heat_maps = []
        self.aug_landmarks = []
        self.occlusions = []
        self.img_paths = []
        self.bounding_boxes = []
        self.train_data = None
        self.validation_data = None
        self.data_size = data_size
        self.landmark_size = landmark_size
        self.color = color
        self.img_size = img_size if img_size is not None else [512, 512]

    def process(self, img_root, img_paths, bounding_boxes, print_debug=False):
        """Whole process"""
        logger("preparing data")
        self._prepare_data(img_root=img_root, img_paths=img_paths,
                           bounding_boxes=bounding_boxes, print_debug=print_debug)

        logger("loading imgs")
        self._load_imgs(print_debug=print_debug)

        logger("normalizing")
        self._normalize_imgs()

        logger("heat_map generating")
        self._heat_map_gen()

    def _prepare_data(self, img_root, img_paths, bounding_boxes, print_debug=False):
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

    def _load_imgs(self, print_debug):
        """Load imgs"""
        for index in range(self.data_size):
            # load img
            if self.color:
                img = cv2.imread(self.img_paths[index])
            else:
                img = cv2.imread(self.img_paths[index], cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger("{} img read error".format(self.img_paths[index]))
                continue
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
            # if index > 10:
            #     break

    def _normalize_imgs(self):
        # self.faces = self.faces.astype(np.float)
        mean_face = np.mean(self.faces, axis=0)
        self.faces = self.faces - mean_face
        std_face = np.std(self.faces, axis=0)
        self.faces = self.faces / std_face

    def _heat_map_gen(self):
        # s = time.time()
        pool = Pool(20)
        candidates = [{'face': face, 'landmark': landmark, 'landmark_01': True} for [face, landmark] in
                      zip(self.faces, self.aug_landmarks)]
        self.heat_maps = pool.map(heat_map_compute, candidates)
        self.heat_maps = [heat_map_compute({'face': face, 'landmark': landmark, 'landmark_01': True})
                          for [face, landmark] in zip(self.faces, self.aug_landmarks)]

        # end = time.time()
        # logger("spending {} seconds".format(int(end - s)))
        # length = len(self.heat_maps)
        # for index in range(length):
        #     show(self.heat_maps[index])

    def train_validation_split(self, test_size, random_state):
        """Train validation data split"""
        self.train_data, self.validation_data = \
            dataset_split(self.heat_maps, self.occlusions,
                          test_size=test_size, random_state=random_state)

    def save(self, dataset_path, file_name=None):
        """Save data"""
        if file_name is None:
            file_name = "dataset_nimgs={0}_size={1}". \
                format(len(self.train_data) + len(self.validation_data), self.img_size)
            if self.color:
                file_name += "_color={0}".format(self.color)
            file_name += ".npz"

        # delete useless data
        del self.faces
        del self.landmarks
        del self.aug_landmarks

        arrays = {key: value for key, value in self.__dict__.items()
                  if not key.startswith('__') and not callable(key)}
        np.savez(os.path.join(dataset_path, file_name), **arrays)
