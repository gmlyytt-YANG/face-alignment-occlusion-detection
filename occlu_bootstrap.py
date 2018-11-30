#! /usr/bin/python
# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: occlu_bootstrap.py
Author: Yang Li
Date: 2018/11/21 09:41:31
Description: Occlu Program Main Entry
"""
import argparse

from config.init_param import occlu_param
from model_run import OcclusionDetection

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

# occlusion detection
occlu_param['epochs'] = args['epoch']
occlu_param['bs'] = args['batch_size']
occlu_param['init_lr'] = args['init_lr']
occlu_param['model_name'] = "best_model_epochs={}_bs={}_lr={}.h5".format(
    occlu_param['epochs'],
    occlu_param['bs'],
    occlu_param['init_lr'])

occlu_clf = OcclusionDetection()
if args["mode"] == "train":
    occlu_clf.train()
elif args["mode"] == "val_compute":
    occlu_clf.validation_benchmark()
