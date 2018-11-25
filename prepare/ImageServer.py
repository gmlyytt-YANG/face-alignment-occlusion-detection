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
import pickle
from multiprocessing import Pool
import time

from config.init_param import occlu_param
from prepare.utils import read_pts, logger, \
    data_aug, dataset_split, heat_map_compute, show, get_face, draw_landmark, gaussian_noise


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

    def __init__(self, data_size, img_size=None, landmark_size=68, 
                 color=False, save_heatmap=False, print_debug=True):
        self.landmarks = []
        self.faces = []
        self.heat_maps = []
        self.img_paths = []
        self.aug_landmarks = []
        self.occlusions = []
        self.bounding_boxes = []
        self.train_data = None
        self.validation_data = None
        self.data_size = data_size
        self.landmark_size = landmark_size
        self.color = color
        self.save_heatmap = save_heatmap
        self.img_size = img_size if img_size is not None else [512, 512]
        self.print_debug = print_debug

    def process(self, img_paths, bounding_boxes):
        """Whole process"""
        logger("preparing data")
        self._prepare_data(img_paths=img_paths,
                           bounding_boxes=bounding_boxes)

        logger("loading imgs")
        self._load_imgs()

        logger("normalizing")
        self._normalize_imgs()

        if self.save_heatmap:
            logger("heat_map generating")
            self._heat_map_gen()

        logger("balancing")
        self._balance()

    def _prepare_data(self, img_paths, bounding_boxes):
        """Getting data
        :param bounding_boxes:
        :param img_paths:
        :return:
        """
        self.bounding_boxes = bounding_boxes
        for index in range(self.data_size):
            img_path = img_paths[index]
            prefix = img_path.split('.')[0]
            pts_path = prefix + '.pts_occlu'
            landmark = read_pts(pts_path)
            # img = cv2.imread(img_path)
            # draw_landmark(img, landmark)
            self.landmarks.append(landmark)
            self.img_paths.append(img_path)
            if self.print_debug:
                if (index + 1) % 100 == 0:
                    logger("processed {} basic infos".format(index + 1))

    def _load_imgs(self):
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
            face = get_face(img, bbox)
            face_dups, landmark_dups, occlusion_dups = data_aug(face=face,
                                                                pts_data=landmark_normalized,
                                                                img_size=self.img_size,
                                                                color=self.color)
            self.faces.extend(face_dups)
            self.aug_landmarks.extend(landmark_dups)
            self.occlusions.extend(occlusion_dups)
            if self.print_debug:
                if (index + 1) % 100 == 0:
                    logger("processed {} images".format(index + 1))
            if index > 50:
               break

    def _balance(self, balanced_num=None):
        occlu_count = 0
        data_size = len(self.occlusions)
        for index, occlusion in enumerate(self.occlusions):
            if np.sum(occlusion) > 0:
                occlu_count += 1
        occlu_ratio = float(occlu_count) / data_size
        logger("occlu_ratio is {}".format(occlu_ratio))
        balanced_num = int(float(1) / occlu_ratio) if balanced_num is None else balanced_num
        occlusions_add = []
        imgs_add = []
        for index in range(len(self.occlusions)):
            if np.sum(self.occlusions[index]) > 0:
                if self.save_heatmap:
                    for num in range(balanced_num):
                        imgs_add.append(gaussian_noise(self.heat_maps[index]))
                        occlusions_add.append(self.occlusions[index])
                else:
                    for num in range(balanced_num):
                        imgs_add.append(gaussian_noise(self.faces[index]))
                        occlusions_add.append(self.occlusions[index])
            if self.print_debug and (index + 1) % 100 == 0:
                logger("processed {} images".format(index + 1))
        self.occlusions.extend(occlusions_add)
        if self.save_heatmap:
            self.heat_maps.extend(imgs_add)
            logger("len heat_map is {}, len occlusion is {}"
                .format(len(self.heat_maps), len(self.occlusions)))
        else:
            self.faces.extend(imgs_add)
            logger("len heat_map is {}, len occlusion is {}"
                .format(len(self.faces), len(self.occlusions)))
    
    def _normalize_imgs(self):
        # self.faces = self.faces.astype(np.float)
        mean_face = np.mean(self.faces, axis=0)
        self.faces = self.faces - mean_face
        std_face = np.std(self.faces, axis=0)
        self.faces = self.faces / std_face
        self.faces = np.multiply(self.faces, 255).astype(int)

    def _heat_map_gen(self):
        # pool = Pool(3)
        # candidates = [{'face': face, 
        #                'landmark': landmark, 
        #                'landmark_01': True, 
        #                'radius': 16} 
        #               for [face, landmark] in zip(self.faces, self.aug_landmarks)]
        # self.heat_maps = pool.map(heat_map_compute, candidates)
        for index in range(len(self.faces)):
            face = self.faces[index]
            landmark = self.aug_landmarks[index]
            self.heat_maps.append(heat_map_compute({'face': face, 
						    'landmark': landmark, 
						    'landmark_01': True, 
						    'radius': occlu_param['radius']}))
            if self.print_debug and (index + 1) % 100 == 0:
                logger("generated {} heatmaps".format(index + 1))

    def train_validation_split(self, test_size, random_state):
        """Train validation data split"""
        if self.save_heatmap:
            self.train_data, self.validation_data = \
                dataset_split(self.heat_maps, self.occlusions,
                              test_size=test_size, random_state=random_state)
        else:
            self.train_data, self.validation_data = \
                dataset_split(self.faces, self.occlusions,
                              test_size=test_size, random_state=random_state)

    def _save_core(self, data_base, dataset_path, mode, print_debug):
        dataset = data_base['data']
        labels = data_base['label']

        data_path = os.path.join(dataset_path, mode)
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        for index in range(len(dataset)):
            data_name = "{}.pkl".format(index)
            data = {
                'image': dataset[index],
                'label': [int(i) for i in labels[index]]
            }
            f_data = open(os.path.join(data_path, data_name), 'wb')
            pickle.dump(data, f_data)
            f_data.close()
            if print_debug and (index + 1) % 500 == 0:
                logger("saved {} data".format(index + 1))

    def save(self, dataset_path):
        """Save data"""
        self._save_core(self.train_data, dataset_path, "train", self.print_debug)
        self._save_core(self.validation_data, dataset_path, "validation", self.print_debug)
