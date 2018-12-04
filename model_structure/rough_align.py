#! /usr/bin/python
# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: rough_align.py
Author: Yang Li
Date: 2018/11/10 17:43:31
Description: Rough Face Alignment
"""

from keras.models import load_model
from model_structure.base_model import *
from ml import *


class FaceAlignmentRough(Model, object):
    def __init__(self):
        train_dir, val_dir = self._load_data()
        super(FaceAlignmentRough, self).__init__(
            lr=face_alignment_rough_param['init_lr'],
            epochs=face_alignment_rough_param['epochs'],
            bs=face_alignment_rough_param['bs'],
            loss=landmark_loss,
            metrics=["accuracy"],
            steps_per_epochs=len(os.listdir(train_dir)) // (occlu_param['bs'] * 6),
            train_dir=train_dir,
            val_dir=val_dir,
            classes=data_param['landmark_num'] * 2
        )

    def val_compute(self, val_load, ext_lists, label_ext, normalizer=None, gpu_ratio=0.5):
        # set gpu usage
        set_gpu(ratio=gpu_ratio)

        # load model
        model = load_model(
            os.path.join(data_param['model_dir'], data_param['model_name']))

        # load data
        val_data, val_labels = val_load(data_dir=self.val_dir,
                                        ext_lists=ext_lists,
                                        label_ext=label_ext,
                                        normalizer=normalizer,
                                        print_debug=self.print_debug)
