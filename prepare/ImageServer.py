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

import pickle

from config.init_param import occlu_param
from prepare.utils import *


class ImageServer(object):
    """Image Source Server

    Attributes:
        landmarks: annotations
        faces:
        heat_maps:
        img_paths: img src path
        aug_landmarks:
        occlusions: occlusion of landmarks
        bounding_boxes: bbox list
        names:
        train_data: list obj, [dict{"data", "label"}]
        validation_data: list obj, [dict{"data", "label"}]
        data_size:
        landmark_size: landmark num
        color:
        img_size: default size is [512, 512]
        save_heatmap: whether to save heat_map or face
        print_debug:
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
        self.names = []
        self.train_data = None
        self.validation_data = None
        self.data_size = data_size
        self.landmark_size = landmark_size
        self.color = color
        self.img_size = img_size if img_size is not None else [512, 512]
        self.save_heatmap = save_heatmap
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
                if (index + 1) % 500 == 0:
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
            face_dups, landmark_dups, occlusion_dups, name_dups = data_aug(face=face,
                                                                           pts_data=landmark_normalized,
                                                                           img_size=self.img_size,
                                                                           name=self.img_paths[index])
            self.faces.extend(face_dups)
            self.aug_landmarks.extend(landmark_dups)
            self.occlusions.extend(occlusion_dups)
            self.names.extend(name_dups)
            if self.print_debug:
                if (index + 1) % 500 == 0:
                    logger("processed {} images".format(index + 1))
            # if index > 50:
            #     break

    def _normalize_imgs(self):
        """Whiten dataset"""
        # self.faces = self.faces.astype(np.float)
        mean_face = np.mean(self.faces, axis=0)
        self.faces = self.faces - mean_face
        std_face = np.std(self.faces, axis=0)
        self.faces = self.faces / std_face
        self.faces = np.multiply(self.faces, 255).astype(int)

    def _heat_map_gen(self):
        """Generate heat map of each of faces"""
        # multi-thread
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
            if self.print_debug and (index + 1) % 500 == 0:
                logger("generated {} heatmaps".format(index + 1))

    def _balance(self, balanced_num=None):
        """Balance dataset
        Increase occlusion objs by (balanced_num + 1) times

        :param balanced_num: required balanced_num to increase nums of occlusion objs
        """
        occlu_count = 0
        data_size = len(self.occlusions)
        for index, occlusion in enumerate(self.occlusions):
            if np.sum(occlusion) > 0:
                occlu_count += 1
        occlu_ratio = float(occlu_count) / data_size
        balanced_num = int(float(1) / occlu_ratio) if balanced_num is None else balanced_num
        occlusions_add = []
        imgs_add = []
        names_add = []
        for index in range(len(self.occlusions)):
            if np.sum(self.occlusions[index]) > 0:
                for num in range(balanced_num):
                    if self.save_heatmap:
                        imgs_add.append(gaussian_noise(self.heat_maps[index]))
                    else:
                        imgs_add.append(gaussian_noise(self.faces[index]))
                    occlusions_add.append(self.occlusions[index])
                    names_add.append(add_postfix(self.names[index], "_occlu_aug_{}".format(num)))
            if self.print_debug and (index + 1) % 500 == 0:
                logger("data aug phase 2 processed {} images".format(index + 1))
        self.occlusions.extend(occlusions_add)
        if self.save_heatmap:
            self.heat_maps.extend(imgs_add)
        else:
            self.faces.extend(imgs_add)
        self.names.extend(names_add)
        logger("length of imgs and occlusions is {}".format(len(self.occlusions)))

    def train_validation_split(self, test_size, random_state):
        """Train validation data split"""
        if self.save_heatmap:
            data_base = self.heat_maps
        else:
            data_base = self.faces
        data_base = [(data, name) for (data, name) in zip(data_base, self.names)]
        self.train_data, self.validation_data = \
            dataset_split(data_base, self.occlusions,
                          test_size=test_size, random_state=random_state)

    @staticmethod
    def _save_core(data_base, dataset_path, mode, print_debug):
        dataset = data_base['data']
        labels = data_base['label']

        data_path = os.path.join(dataset_path, mode)
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        for index in range(len(dataset)):
            data_name = "{}.pkl".format(index)
            data = {
                'image_name': dataset[index],
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
