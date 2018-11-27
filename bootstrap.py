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
Description: Program Main Entry
"""
import argparse
import os

from config.init_param import occlu_param
from occlusion_detection import OcclusionDetection
from prepare.utils import load_imgs, logger

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
args = vars(ap.parse_args())

occlu_param['epochs'] = args['epoch']
occlu_param['bs'] = args['batch_size']
occlu_param['init_lr'] = args['init_lr']
occlu_param['model_name'] = "best_model_epochs={}_bs={}_lr={}.h5".format(
    occlu_param['epochs'],
    occlu_param['bs'],
    occlu_param['init_lr'])

# occlusion detection
occlu_clf = OcclusionDetection()
if args["mode"] == "train":
    occlu_clf.data_pre()
    # occlu_clf.train()
    pass
elif args["mode"] == "val_compute":
    occlu_clf.validation_benchmark()

# mat_file = os.path.join(occlu_param['img_root_dir'], 'raw_300W_release.mat')
# for face in load_imgs(occlu_param['img_root_dir'],
#                       mat_file_name=mat_file,
#                       total=False,
#                       chosed=[6, 7]):
#     OcclusionDetection.classify(face, need_to_normalize=True)
