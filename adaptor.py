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
import time
import keras.backend as K

from config.init_param import data_param, occlu_param, fap_param
from config.parse_param import parse_param
from model_structure.align_v1 import FaceAlignment
from model_structure.occlu_detect import OcclusionDetection
from ml import load_config
from utils import get_filenames
from utils import heat_map_compute
from utils import load_basic_info
from utils import load_imgs_labels
from utils import logger
from utils import set_gpu

# load config
normalizer, mean_shape = load_config()
set_gpu(ratio=0.5)

# parse parameter
ap = argparse.ArgumentParser()
ap.add_argument('-e1', '--epochs1', type=int, default=75, help='epochs1 of face alignment')
ap.add_argument('-bs1', '--batch_size1', type=int, default=32, help='batch size of face alignment')
ap.add_argument('-lr1', '--init_lr1', type=float, default=1e-3, help='learning rate of face alignment')
ap.add_argument('-mt1', '--model_type1', type=str, default='vgg16_clf', help='model type of fap')
ap.add_argument('-ln1', '--loss_name1', type=str, default='landmark_loss of fap')
ap.add_argument('-c1', '--content1', type=str, help='description of fap model')
ap.add_argument('-e2', '--epochs2', type=int, default=75, help='epochs2 of occlu detection')
ap.add_argument('-bs2', '--batch_size2', type=int, default=32, help='batch size of occlu detection')
ap.add_argument('-lr2', '--init_lr2', type=float, default=1e-3, help='learning rate of occlu detection')
ap.add_argument('-mt2', '--model_type2', type=str, default='vgg16_clf', help='model type of occlu')
ap.add_argument('-ln2', '--loss_name2', type=str, default='landmark_loss of occlu')
ap.add_argument('-c2', '--content2', type=str, help='description of occlu model')
ap.add_argument('-le', '--label_ext', type=str, default='.pts', help='label ext type')
ap.add_argument('-f', '--feature', type=str, default='face', help='input feature type')
args = ap.parse_args()
args = vars(args)

# load parameter
epochs1 = args['epochs1']
bs1 = args['batch_size1']
lr1 = args['init_lr1']
mt1 = args['model_type1']
ln1 = args['loss_name1']
content1 = args['content1']
model_name1 = 'best_model_epochs={}_bs={}_lr={}_des={}.h5'.format(epochs1, bs1, lr1, content1)
epochs2 = args['epochs2']
bs2 = args['batch_size2']
lr2 = args['init_lr2']
mt2 = args['model_type2']
ln2 = args['loss_name2']
content2 = args['content2']
model_name2 = 'best_model_epochs={}_bs={}_lr={}_des={}.h5'.format(epochs2, bs2, lr2, content2)
label_ext = args['label_ext']
feature = args['feature']
model_structure1, loss1, loss_compute1, model1 = \
    parse_param(model_type=mt1, loss_name=ln1, model_name=model_name1)
model_structure2, loss2, loss_compute2, model2 = \
    parse_param(model_type=mt2, loss_name=ln2, model_name=model_name2)


def get_pipe_data(img):
    """Get weighted landmark based on rough face alignment
       and occlusion detection"""
    start_time = time.time()
    prediction = FaceAlignment.test(img=img, mean_shape=mean_shape,
                                    normalizer=normalizer, model=model1)
    img = heat_map_compute(face=img, landmark=prediction,
                           landmark_is_01=False, img_color=True,
                           radius=occlu_param['radius'])
    occlu_ratio = OcclusionDetection.test(img=img, landmark=prediction, is_heat_map=True)
    end_time = time.time()
    logger("time of processing one img is {}".format(end_time - start_time))
    return np.array(prediction).flatten(), \
           np.expand_dims(np.array(occlu_ratio), axis=1).flatten(), end_time - start_time


def pipe_train_val(data_dir, img_ext_lists=None):
    """Pipeline of saving imgs in train and val dir"""
    img_name_list, label_name_list = \
        get_filenames(data_dir=data_dir, img_ext_lists=img_ext_lists, label_ext=label_ext)

    count = 0
    time_total = 0.0
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
    logger("average speed for processing is {} fps".format(float(count) / time_total))


def pipe_test():
    faces, labels = load_imgs_labels(img_root=data_param['img_root_dir'],
                                     img_size=data_param['img_size'],
                                     normalizer=normalizer,
                                     chosen=range(3148, 3837))
    img_paths, _ = load_basic_info('raw_300W_release.mat', data_param['img_root_dir'])
    count = 0
    time_total = 0.0
    for (img, landmark) in zip(faces, labels):
        prediction, occlu_ratio, time_pass = get_pipe_data(img)
        delta = np.concatenate((landmark, prediction, occlu_ratio))
        np.savetxt(os.path.splitext(img_paths[count])[0] + '.wdpts', delta, fmt='%.10f')
        if data_param['print_debug'] and (count + 1) % 100 == 0:
            logger('saved {} wdpts'.format(count + 1))
        time_total += time_pass
    logger("average speed for processing is {} fps".format(float(count) / time_total))


if __name__ == "__main__":
    logger("save training data")
    train_data_dir = os.path.join(data_param['train_dir'], feature)
    val_data_dir = os.path.join(data_param['val_dir'], feature)
    pipe_train_val(train_data_dir, img_ext_lists=data_param['img_ext'])
    pipe_train_val(val_data_dir, img_ext_lists=data_param['img_ext'])

    logger("save test data")
    pipe_test()

    # close session
    K.clear_session()
