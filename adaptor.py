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
import time
from keras.models import load_model
import keras.backend as K

from config.init_param import data_param, occlu_param, face_alignment_rough_param
from ml import landmark_loss
from ml import load_config
from utils import get_filenames
from utils import heat_map_compute
from utils import load_basic_info
from utils import load_rough_imgs_labels_core
from utils import logger
from utils import set_gpu

# parse parameter
ap = argparse.ArgumentParser()
ap.add_argument('-e1', '--epoch1', type=int, default=75, help='epochs1 of face alignment')
ap.add_argument('-bs1', '--batch_size1', type=int, default=32, help='batch size of face alignment')
ap.add_argument('-lr1', '--init_lr1', type=float, default=1e-3, help='learning rate of face alignment')
ap.add_argument('-e2', '--epoch2', type=int, default=75, help='epochs2 of occlu detection')
ap.add_argument('-bs2', '--batch_size2', type=int, default=32, help='batch size of occlu detection')
ap.add_argument('-lr2', '--init_lr2', type=float, default=1e-3, help='learning rate of occlu detection')
args = ap.parse_args()
args = vars(args)

# load parameter
face_alignment_rough_param['epochs'] = args['epoch1']
face_alignment_rough_param['bs'] = args['batch_size1']
face_alignment_rough_param['init_lr'] = args['init_lr1']
face_alignment_rough_param['model_name'] = 'best_model_epochs={}_bs={}_lr={}_rough.h5'.format(
    face_alignment_rough_param['epochs'],
    face_alignment_rough_param['bs'],
    face_alignment_rough_param['init_lr'])
occlu_param['epochs'] = args['epoch2']
occlu_param['bs'] = args['batch_size2']
occlu_param['init_lr'] = args['init_lr2']
occlu_param['model_name'] = 'best_model_epochs={}_bs={}_lr={}_occlu.h5'.format(
    occlu_param['epochs'],
    occlu_param['bs'],
    occlu_param['init_lr'])

# load mean_shape and normalizer
normalizer, mean_shape = load_config()

# gpu related
set_gpu(ratio=0.5)
model_rough = load_model(os.path.join(data_param['model_dir'], face_alignment_rough_param['model_name']),
                         {'landmark_loss': landmark_loss})
model_occlu = load_model(os.path.join(data_param['model_dir'], occlu_param['model_name']))

from model_structure.occlu_detect import OcclusionDetection
from model_structure.rough_align import FaceAlignment


def get_pipe_data(img):
    """Get weighted landmark based on rough face alignment and occlusion detection"""
    start_time = time.time()
    prediction = FaceAlignment.test(img=img, mean_shape=mean_shape, normalizer=normalizer)
    img = heat_map_compute(face=img, landmark=prediction,
                           landmark_is_01=False, img_color=True, radius=occlu_param['radius'])
    occlu_ratio = OcclusionDetection.test(img=img, landmark=prediction, is_heat_map=True)
    end_time = time.time()
    # logger("time of processing one img is {}".format(end_time - start_time))

    return np.array(prediction).flatten(), np.expand_dims(np.array(occlu_ratio),
                                                          axis=1).flatten(), end_time - start_time


def pipe(data_dir, face=False, chosen=range(1)):
    """Pipeline of saving imgs"""
    time_total = 0.0
    if face:
        img_name_list, label_name_list = \
            get_filenames(data_dir=[data_dir],
                          listext=["*_face.png", "*_face.jpg"],
                          label_ext=".pts")

        count = 0
        for img_path, label_path in zip(img_name_list, label_name_list):
            img = cv2.imread(img_path)
            landmark = np.genfromtxt(label_path)
            prediction, occlu_ratio, time_pass = get_pipe_data(img)
            delta = np.concatenate((landmark, prediction, occlu_ratio))
            np.savetxt(os.path.splitext(img_path)[0] + '.wdpts', delta, fmt='%.10f')
            count += 1
            if data_param['print_debug'] and count % 500 == 0:
                logger('saved {} wdpts'.format(count))
            time_total += time_pass

    else:
        img_paths, bboxes = load_basic_info('raw_300W_release.mat', data_dir)
        count = 0
        for index in chosen:
            img, landmark = load_rough_imgs_labels_core(img_path=img_paths[index],
                                                        bbox=bboxes[index],
                                                        img_size=data_param['img_size'])
            prediction, occlu_ratio, time_pass = get_pipe_data(img)
            delta = np.concatenate((landmark, prediction, occlu_ratio))
            np.savetxt(os.path.splitext(img_paths[index])[0] + '.wdpts', delta, fmt='%.10f')
            if data_param['print_debug'] and (index + 1) % 100 == 0:
                logger('saved {} wdpts'.format(index + 1))
            count = index + 1
            time_total += time_pass
    logger("average speed for processing is {} fps".format(float(count) / time_total))


if __name__ == "__main__":
    # save data
    logger("save training data")
    pipe(os.path.join(data_param['data_save_dir'], 'train'), face=True)
    logger("save val data")
    pipe(os.path.join(data_param['data_save_dir'], 'val'), face=True)
    logger("save test data")
    pipe(data_param['img_root_dir'], face=False, chosen=range(3148, 3837))

    # close session
    K.clear_session()
