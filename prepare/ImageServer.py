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

from sklearn.model_selection import train_test_split

from config.init_param import *
from ml import *


class ImageServer(object):
    """Image Source Server

    Attributes:
        landmarks: annotations
        faces:
        heat_maps:
        img_paths: img src path
        landmarks: original landmarks, which will be deleted after getting aug_landmarks
        aug_landmarks: augmented landmarks
        occlusions: occlusion of landmarks
        bboxes: bbox list
        names:
        train_data: list obj, [dict{"data", "label"}]
        validation_data: list obj, [dict{"data", "label"}]
        data_size:
        landmark_size: landmark num
        color:
        img_size: default size is 112
        print_debug:
    """

    def __init__(self, img_size=112, landmark_size=68,
                 color=False, print_debug=True):
        self.data_size = 0
        self.landmarks = []
        self.faces = []
        self.heat_maps = []
        self.img_paths = []
        self.aug_landmarks = []
        self.occlusions = []
        self.bboxes = []
        self.names = []
        self.train_data = None
        self.validation_data = None
        self.landmark_size = landmark_size
        self.color = color
        self.img_size = img_size
        self.print_debug = print_debug

    def process(self, img_paths, bboxes):
        """Whole process"""
        logger("preparing data")
        self._prepare(img_paths, bboxes)

        logger("loading imgs")
        self._load_imgs()

        logger("normalizing")
        self._normalize_imgs()

        logger("heat_map generating")
        self._heat_map_gen()

        logger("balancing")
        self._balance()

        # splitting
        logger("train validation splitting")
        self._train_val_split()

    def _prepare(self, img_paths, bboxes):
        """Getting data
        :param img_paths:
        :param bboxes:

        :return:
        """
        self.bboxes = bboxes
        for index in range(len(img_paths)):
            img_path = img_paths[index]
            landmark = np.genfromtxt(os.path.splitext(img_path)[0] + ".opts",
                                     delimiter=" ", max_rows=self.landmark_size)
            # img = cv2.imread(img_path)
            # draw_landmark(img, landmark)
            self.landmarks.append(landmark)
            self.img_paths.append(img_path)
            if self.print_debug and (index + 1) % 500 == 0:
                logger("processed {} basic infos".format(index + 1))
            # if (index + 1) >= 10:
            #     break

    def _load_imgs(self):
        """Load imgs"""
        for index in range(len(self.landmarks)):
            # load img
            if self.color:
                img = cv2.imread(self.img_paths[index])
            else:
                img = cv2.imread(self.img_paths[index], cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger("{} img read error".format(self.img_paths[index]))
                continue
            bbox = [int(_) for _ in self.bboxes[index]]

            # normalize landmark
            landmark = self.landmarks[index]
            landmark_normalized = normalize_data(landmark)

            # data augment
            face = cv2.resize(get_face(img, bbox), (self.img_size, self.img_size))
            name = self.img_paths[index].lstrip(data_param['img_root_dir']).replace("/", "_")
            faces, landmarks, occlusions, names = \
                data_aug(face=face, landmark=landmark_normalized, name=name)
            self.faces.extend(faces)
            self.aug_landmarks.extend(landmarks)
            self.occlusions.extend(occlusions)
            self.names.extend(names)

            if self.print_debug and (index + 1) % 500 == 0:
                logger("processed {} images".format(index + 1))

        self.data_size = len(self.occlusions)

        del self.landmarks

    def _normalize_imgs(self):
        normalizer = StdMinMaxScaler()
        self.faces = normalizer.fit_transform(self.faces)
        # for face in self.faces:
        #     show(face)
        create_dir(data_param['normalizer_dir'])
        np.savez(os.path.join(data_param['normalizer_dir'], "normalizer.npz"))

    def _heat_map_gen(self):
        """Generate heat map of each of faces"""
        for index in range(self.data_size):
            face = self.faces[index]
            landmark = self.aug_landmarks[index]
            self.heat_maps.append(heat_map_compute(face, landmark,
                                                   landmark_is_01=True,
                                                   img_color=self.color,
                                                   radius=occlu_param['radius']))
            if self.print_debug and (index + 1) % 500 == 0:
                logger("generated {} heatmaps".format(index + 1))

    def _balance(self, balanced_num=None):
        """Balance dataset
        Increase occlusion objs by (balanced_num + 1) times

        :param balanced_num: required balanced_num to increase nums of occlusion objs
        """
        count = 0
        for index in range(self.data_size):
            if np.sum(self.occlusions[index]) > 0:
                count += 1
        ratio = float(count) / self.data_size
        balanced_num = int(float(1) / ratio) if balanced_num is None else balanced_num
        occlusions_add = []
        heatmaps_add = []
        faces_add = []
        names_add = []
        landmarks_add = []
        for index in range(len(self.occlusions)):
            if np.sum(self.occlusions[index]) > 0:
                for num in range(balanced_num):
                    heatmap = gaussian_noise(self.heat_maps[index], color=self.color)
                    heatmaps_add.append(heatmap)
                    face = gaussian_noise(self.faces[index], color=self.color)
                    faces_add.append(face)
                    occlusions_add.append(self.occlusions[index])
                    landmarks_add.append(self.aug_landmarks[index])
                    names_add.append(add_postfix(self.names[index], "_gaussian_{}".format(num)))
            if self.print_debug and (index + 1) % 500 == 0:
                logger("data aug phase 2 processed {} images".format(index + 1))
        self.faces = extend(self.faces, faces_add)
        self.occlusions.extend(occlusions_add)
        self.heat_maps.extend(heatmaps_add)
        self.aug_landmarks.extend(landmarks_add)
        self.names.extend(names_add)
        self.data_size = len(self.occlusions)
        logger("length of imgs and occlusions is {}".format(self.data_size))

    def _split_core(self, x, y, mode, phase):
        data_dir = os.path.join(data_param['data_save_dir'], mode)
        for index in range(len(x)):
            img = x[index][0]
            name = x[index][1]
            landmark = y[index][0] * self.img_size
            occlusion = y[index][1]

            # save data
            img_path = os.path.join(data_dir, add_postfix(name, "_{}".format(phase)))
            cv2.imwrite(img_path, img)
            np.savetxt(os.path.splitext(img_path)[0] + ".pts", landmark, fmt="%.10f")
            np.savetxt(os.path.splitext(img_path)[0] + ".opts", occlusion, fmt="%d")

    def _img_split(self, phase="face"):
        if phase == "face":
            x = [_ for _ in zip(self.faces, self.names)]
        else:
            x = [_ for _ in zip(self.heat_maps, self.names)]
        y = [_ for _ in zip(self.aug_landmarks, self.occlusions)]
        x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True,
                                                            test_size=data_param['test_size'],
                                                            random_state=data_param['random_state'])
        self._split_core(x_train, y_train, mode="train", phase=phase)
        self._split_core(x_test, y_test, mode="val", phase=phase)

    def _train_val_split(self):
        """Train validation data split"""
        # init
        data_dir = os.path.join(data_param['data_save_dir'], "train")
        create_dir(data_dir)
        remove_content(data_dir)
        data_dir = os.path.join(data_param['data_save_dir'], "val")
        create_dir(data_dir)
        remove_content(data_dir)

        # save data
        self._img_split(phase="face")
        self._img_split(phase="heatmap")
