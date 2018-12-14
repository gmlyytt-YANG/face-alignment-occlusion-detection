#! /usr/bin/python
# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: train.py
Author: Yang Li
Date: 2018/11/21 09:41:31
Description: Main Entry of Training
"""
import argparse
import os
import keras.backend as K
from keras.models import load_model
import socket

from config.init_param import data_param, occlu_param, far_param, fap_param
from config.parse_param import parse_param
from data_gen import train_data_feed
from data_gen import val_data_feed
from ml import load_config
from model_structure.align_v1 import FaceAlignment
from model_structure.occlu_detect import OcclusionDetection
from utils import count_file
from utils import logger
from utils import set_gpu

# load config
normalizer, mean_shape = load_config()
host_name = socket.gethostname()
if host_name == 'KB249-workstation':
    set_gpu(ratio=0.3)
else:
    set_gpu(ratio=0.4)

# load parameter
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--content', type=str, help='description of model')
ap.add_argument('-e', '--epochs', type=int, default=75, help='epochs of training')
ap.add_argument('-bs', '--batch_size', type=int, default=32, help='batch size of training')
ap.add_argument('-lr', '--init_lr', type=float, default=1e-3, help='learning rate')
ap.add_argument('-m', '--mode', type=str, default='val_compute', help='mode of ML')
ap.add_argument('-p', '--phase', type=str, default='rough', help='phase of pipeline')
ap.add_argument('-f', '--feature', type=str, default='face', help='input feature type')
ap.add_argument('-le', '--label_ext', type=str, default='.pts', help='label ext type')
ap.add_argument('-mt', '--model_type', type=str, default='vgg16_clf', help='model type')
ap.add_argument('-ln', '--loss_name', type=str, default='landmark_loss')
args, unknown = ap.parse_known_args()
args = vars(args)

# parse parameter
label_ext = args['label_ext']
feature = args['feature']
model_type = args['model_type']
loss_name = args['loss_name']
epochs = args['epochs']
bs = args['batch_size']
lr = args['init_lr']
content = args['content']
model_name = 'best_model_epochs={}_bs={}_lr={}_des={}.h5'.format(epochs, bs, lr, content)
model_structure, loss, loss_compute = \
    parse_param(model_type=model_type, loss_name=loss_name)
train_data_dir = os.path.join(data_param['train_dir'], feature)
val_data_dir = os.path.join(data_param['val_dir'], feature)
train_num = count_file([train_data_dir], data_param['img_ext'])

# face alignment rough
if args['phase'] == 'rough':
    face_align_rgr = FaceAlignment(lr=lr, epochs=epochs, bs=bs,
                                   model_name=model_name, classes=data_param['landmark_num'] * 2,
                                   loss=loss, train_num=train_num, esm=fap_param['es_monitor'])
    weight_path = os.path.join(far_param['weight_path'], far_param['weight_name'])
    train_vars = {'data_dir': train_data_dir, 'img_ext_lists': data_param['img_ext'],
                  'label_ext': label_ext, 'flatten': True}
    val_vars = {'data_dir': val_data_dir, 'img_ext_lists': data_param['img_ext'],
                'label_ext': label_ext, 'flatten': True}
    logger("epochs: {}, bs: {}, lr: {}".format(epochs, bs, lr))
    face_align_rgr.train(model_structure=model_structure, train_load=train_data_feed, train_vars=train_vars,
                         val_load=val_data_feed, val_vars=val_vars, weight_path=weight_path)

# occlusion detection
if args['phase'] == 'occlu':
    # training
    occlu_clf = OcclusionDetection(lr=lr, epochs=epochs, bs=bs, model_name=model_name,
                                   loss='binary_crossentropy', train_num=train_num, esm=occlu_param['es_monitor'])
    weight_path = os.path.join(occlu_param['weight_path'], occlu_param['weight_name'])
    train_vars = {'data_dir': train_data_dir, 'img_ext_lists': data_param['img_ext'],
                  'label_ext': label_ext, 'flatten': False}
    val_vars = {'data_dir': val_data_dir, 'img_ext_lists': data_param['img_ext'],
                'label_ext': label_ext, 'flatten': False}
    logger("epochs: {}, bs: {}, lr: {}".format(epochs, bs, lr))
    val_data, val_labels = occlu_clf.train(model_structure=model_structure, train_load=train_data_feed,
                                           train_vars=train_vars, val_load=val_data_feed,
                                           val_vars=val_vars, weight_path=weight_path)
    # evaluating
    if loss_name != 'no':
        model = load_model(os.path.join(data_param['model_dir'], model_name), {loss_name: loss})
    else:
        model = load_model(os.path.join(data_param['model_dir'], model_name))
    if feature == 'heatmap':
        val_vars = {'data_dir': val_data_dir, 'img_ext_lists': data_param['img_ext'],
                    'label_ext': '.pts', 'flatten': True}
        _, landmarks = val_data_feed(data_dict=val_vars)
        data = zip(val_data, landmarks)
        is_heatmap = True
    else:
        data = val_vars
        is_heatmap = False
    OcclusionDetection.val_compute(data, val_labels, model=model, is_heatmap=is_heatmap)

# face precise alignment
if args['phase'] == 'precise':
    val_data_dir = data_param['val_dir']
    face_align_rgr = FaceAlignment(lr=lr, epochs=epochs, bs=bs, model_name=model_name,
                                   classes=data_param['landmark_num'] * 2, esm=fap_param['es_monitor'],
                                   loss=loss, train_num=train_num)
    weight_path = os.path.join(fap_param['weight_path'], fap_param['weight_name'])
    train_vars = {'data_dir': train_data_dir, 'img_ext_lists': data_param['img_ext'],
                  'label_ext': label_ext, 'flatten': False}
    val_vars = {'data_dir': val_data_dir, 'img_ext_lists': data_param['img_ext'],
                'label_ext': label_ext, 'flatten': True}
    logger("epochs: {}, bs: {}, lr: {}".format(epochs, bs, lr))
    face_align_rgr.train(model_structure=model_structure, train_load=train_data_feed, train_vars=train_vars,
                         val_load=val_data_feed, val_vars=val_vars, weight_path=weight_path)

K.clear_session()
