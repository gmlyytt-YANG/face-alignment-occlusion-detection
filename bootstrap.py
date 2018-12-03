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
Description: Occlu Program Main Entry
"""
import argparse

from config.init_param import *
from model_structure.model_run import OcclusionDetection, FaceAlignmentRough
from prepare.data_gen import *
from model_structure.vgg16 import *

# load parameter
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epoch", type=int, default=75,
                help="epochs of training")
ap.add_argument("-bs", "--batch_size", type=int, default=32,
                help="batch size of training")
ap.add_argument("-lr", "--init_lr", type=float, default=1e-3,
                help="learning rate")
ap.add_argument("-m", "--mode", type=str, default="train",
                help="mode of ML")
ap.add_argument("-p", "--phase", type=str, default="rough",
                help="phase of pipeline")
args = vars(ap.parse_args())

# occlusion detection
normalizer = np.load(os.path.join(data_param['normalizer_dir'], "normalizer.npz"))
if args["phase"] == "occlu":
    occlu_param['epochs'] = args['epoch']
    occlu_param['bs'] = args['batch_size']
    occlu_param['init_lr'] = args['init_lr']
    occlu_param['model_name'] = "best_model_epochs={}_bs={}_lr={}_occlu.h5".format(
        occlu_param['epochs'],
        occlu_param['bs'],
        occlu_param['init_lr'])

    # learning
    occlu_clf = OcclusionDetection()
    if args["mode"] == "train":
        occlu_clf.train(model_structure=Vgg16CutFC2(),
                        train_load=train_data_feed,
                        val_load=validation_data_feed,
                        ext_lists=["*_heatmap.png", "*_heatmap.jpg"],
                        label_ext=".opts",
                        gpu_ratio=0.5)
    elif args["mode"] == "val_compute":
        occlu_clf.val_compute(val_load=validation_data_feed,
                              ext_lists=["*_heatmap.png", "*_heatmap.jpg"],
                              label_ext=".opts",
                              gpu_ratio=0.5)

# face alignment rough
if args["phase"] == "rough":
    face_alignment_rough_param['epochs'] = args['epoch']
    face_alignment_rough_param['bs'] = args['batch_size']
    face_alignment_rough_param['init_lr'] = args['init_lr']
    face_alignment_rough_param['model_name'] = "best_model_epochs={}_bs={}_lr={}.h5".format(
        face_alignment_rough_param['epochs'],
        face_alignment_rough_param['bs'],
        face_alignment_rough_param['init_lr'])

    face_align_rgr = FaceAlignmentRough()
    if args["mode"] == "train":
        face_align_rgr.train(model_structure=Vgg16Regress(),
                             train_load=train_data_feed,
                             val_load=validation_data_feed,
                             ext_lists=["*_face.png", "*_face.jpg"],
                             label_ext=".pts",
                             normalizer=normalizer,
                             gpu_ratio=0.5)
