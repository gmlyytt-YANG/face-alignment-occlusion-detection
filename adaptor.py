#! /usr/bin/python
# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: adaptor.py
Author: Yang Li
Date: 2018/12/04 20:04:00
Description: Adaptor for rough face alignment and occlusion detection.
"""
import argparse
import cv2
import numpy as np
import os
import pickle

from config.init_param import data_param, occlu_param, face_alignment_rough_param
from model_structure.rough_align import FaceAlignment
from model_structure.occlu_detect import OcclusionDetection
from ml import landmark_delta_loss
from utils import get_filenames
from utils import heat_map_compute
from utils import load_basic_info
from utils import load_rough_imgs_labels_core
from utils import logger

# load parameter
ap = argparse.ArgumentParser()
ap.add_argument('-h1', '--epoch1', type=int, default=75,
                help='epochs1 of face alignment')
ap.add_argument('-p1', '--batch_size1', type=int, default=32,
                help='batch size of face alignment')
# ap.add_argument('-LR1', '--init_lr1', type=float, default=1e-3,
#                 help='learning rate of face alignment')
# ap.add_argument('-E2', '--epoch2', type=int, default=75,
#                 help='epochs2 of occlu detection')
# ap.add_argument('-BS2', '--batch_size2', type=int, default=32,
#                 help='batch size of occlu detection')
# ap.add_argument('-LR2', '--init_lr2', type=float, default=1e-3,
#                 help='learning rate of occlu detection')
args, unknown = ap.parse_known_args()
args = vars(args)

face_alignment_rough_param['epochs'] = args['epoch1']
face_alignment_rough_param['bs'] = args['batch_size1']
# face_alignment_rough_param['init_lr'] = args['init_lr1']
# face_alignment_rough_param['model_name'] = 'best_model_epochs={}_bs={}_lr={}_rough.h5'.format(
#     face_alignment_rough_param['epochs'],
#     face_alignment_rough_param['bs'],
#     face_alignment_rough_param['init_lr'])
# occlu_param['epochs'] = args['epoch2']
# occlu_param['bs'] = args['batch_size2']
# occlu_param['init_lr'] = args['init_lr2']
# occlu_param['model_name'] = 'best_model_epochs={}_bs={}_lr={}_occlu.h5'.format(
#     occlu_param['epochs'],
#     occlu_param['bs'],
#     occlu_param['init_lr'])

# load mean_shape and normalizer
f_mean_shape = open(os.path.join(data_param['model_dir'], 'mean_shape.pkl'), 'rb')
mean_shape = pickle.load(f_mean_shape)
f_mean_shape.close()
f_normalizer = open(os.path.join(data_param['model_dir'], 'normalizer.pkl'), 'rb')
normalizer = pickle.load(f_normalizer)
f_normalizer.close()


def get_weighted_landmark(img, landmark):
    """Get weighted landmark based on rough face alignment and occlusion detection"""
    prediction = FaceAlignment(loss=landmark_delta_loss).test(img=img,
                                                              mean_shape=mean_shape,
                                                              normalizer=normalizer)
    img = heat_map_compute(face=img,
                           landmark=prediction,
                           landmark_is_01=False,
                           img_color=True,
                           radius=occlu_param['radius'])
    occlu_ratio = OcclusionDetection().test(img=img,
                                            landmark=landmark,
                                            is_heat_map=True)
    delta = (landmark - prediction) * occlu_ratio.T
    left_eye = np.mean(landmark[36:42, :], axis=0)
    right_eye = np.mean(landmark[42:48, :], axis=0)
    pupil_dist = np.sqrt(np.sum((left_eye - right_eye) ** 2))

    return np.concatenate((delta.flatten(), np.array([pupil_dist])))


# load data
def pipe(data_dir, face=False, chosen=range(1)):
    if face:
        img_name_list, label_name_list = \
            get_filenames(data_dir=[data_dir],
                          listext=["*_face.png", "*_face.jpg"],
                          label_ext=[".pts"])

        for img_path, label_path in zip(img_name_list, label_name_list):
            img = cv2.imread(img_path)
            landmark = np.genfromtxt(label_path, skip_header=3, skip_footer=1)
            delta = get_weighted_landmark(img, landmark)
            np.savetxt(os.path.splitext(img_path)[0] + ".wdpts", delta, fmt='%.10f')
    else:
        img_paths, bboxes = load_basic_info('raw_300W_release.mat', data_dir)
        for index in chosen:
            img, landmark = load_rough_imgs_labels_core(img_path=img_paths[index],
                                                        bbox=bboxes[index],
                                                        img_size=data_param['img_size'])
            delta = get_weighted_landmark(img, landmark)
            np.savetxt(os.path.splitext(img_paths[index])[0] + ".wdpts", delta, fmt='%.10f')


if __name__ == "__main__":
    # train data
    logger("save training data")
    pipe(os.path.join(data_param['data_save_dir'], 'train'), face=True)

    # val data
    logger("save val data")
    pipe(os.path.join(data_param['data_save_dir'], 'val'), face=True)

    # test data
    logger("save test data")
    pipe(data_param['img_root_dir'], face=False, chosen=range(3148, 3837))
