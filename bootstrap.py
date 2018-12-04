#! /usr/bin/python
# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: bootstrap.py
Author: Yang Li
Date: 2018/11/21 09:41:31
Description: Main Entry
"""
import argparse
import pickle
import os
from keras.models import load_model

from config.init_param import data_param, occlu_param, face_alignment_rough_param
from model_structure.occlu_detect import OcclusionDetection
from model_structure.rough_align import FaceAlignmentRough
from model_structure.vgg16 import Vgg16Regress, Vgg16CutFC2
from prepare.data_gen import train_data_feed, val_data_feed
from utils import load_rough_imgs_labels
from utils import load_rough_imgs_occlus
from ml import metric_compute

# load parameter
ap = argparse.ArgumentParser()
ap.add_argument('-e', '--epoch', type=int, default=75,
                help='epochs of training')
ap.add_argument('-bs', '--batch_size', type=int, default=32,
                help='batch size of training')
ap.add_argument('-lr', '--init_lr', type=float, default=1e-3,
                help='learning rate')
ap.add_argument('-m', '--mode', type=str, default='val_compute',
                help='mode of ML')
ap.add_argument('-p', '--phase', type=str, default='rough',
                help='phase of pipeline')
args = vars(ap.parse_args())

# load mean_shape and normalizer 
f_mean_shape = open(os.path.join(data_param['model_dir'], 'mean_shape.pkl'), 'rb')
mean_shape = pickle.load(f_mean_shape)
f_mean_shape.close()
f_normalizer = open(os.path.join(data_param['model_dir'], 'normalizer.pkl'), 'rb')
normalizer = pickle.load(f_normalizer)
f_normalizer.close()

# occlusion detection
if args['phase'] == 'occlu':
    occlu_param['epochs'] = args['epoch']
    occlu_param['bs'] = args['batch_size']
    occlu_param['init_lr'] = args['init_lr']
    occlu_param['model_name'] = 'best_model_epochs={}_bs={}_lr={}_occlu.h5'.format(
        occlu_param['epochs'],
        occlu_param['bs'],
        occlu_param['init_lr'])

    # learning
    occlu_clf = OcclusionDetection()
    if args['mode'] == 'train':
        occlu_clf.train(model_structure=Vgg16CutFC2(),
                        train_load=train_data_feed,
                        val_load=val_data_feed,
                        ext_lists=['*_heatmap.png', '*_heatmap.jpg'],
                        label_ext='.opts',
                        gpu_ratio=0.5)
    elif args['mode'] == 'val_compute':
        occlu_clf.val_compute(val_load=val_data_feed,
                              ext_lists=['*_heatmap.png', '*_heatmap.jpg'],
                              label_ext='.opts',
                              gpu_ratio=0.5)
    elif args['mode'] == 'test':
        model = load_model(
            os.path.join(data_param['model_dir'], occlu_param['model_name']))
        faces, labels = load_rough_imgs_occlus(
            img_root=data_param['img_root_dir'],
            mat_file_name='raw_300W_release.mat',
            img_size=data_param['img_size'],
            chosen=range(3148, -1)
        )
        predictions = []
        for face, label in zip(faces, labels):
            prediction = occlu_clf.test(model=model,
                                        img=face,
                                        landmark=label,
                                        is_heat_map=True,
                                        binary_output=True)
            predictions.append(prediction)
        metric_compute(labels, predictions)


# face alignment rough
if args['phase'] == 'rough':
    face_alignment_rough_param['epochs'] = args['epoch']
    face_alignment_rough_param['bs'] = args['batch_size']
    face_alignment_rough_param['init_lr'] = args['init_lr']
    face_alignment_rough_param['model_name'] = 'best_model_epochs={}_bs={}_lr={}.h5'.format(
        face_alignment_rough_param['epochs'],
        face_alignment_rough_param['bs'],
        face_alignment_rough_param['init_lr'])

    face_align_rgr = FaceAlignmentRough()
    if args['mode'] == 'train':
        face_align_rgr.train(model_structure=Vgg16Regress(),
                             train_load=train_data_feed,
                             val_load=val_data_feed,
                             ext_lists=['*_face.png', '*_face.jpg'],
                             label_ext='.pts',
                             mean_shape=mean_shape,
                             normalizer=normalizer,
                             gpu_ratio=0.5)
    if args['mode'] == 'val_compute':
        faces, labels = load_rough_imgs_labels(img_root=data_param['img_root_dir'],
                                               mat_file_name='raw_300W_release.mat',
                                               img_size=data_param['img_height'],
                                               normalizer=normalizer,
                                               chosen=range(3148, -1))
        face_align_rgr.val_compute(imgs=faces, labels=labels, gpu_ratio=0.5)
