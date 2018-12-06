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

from config.init_param import face_alignment_rough_param
from model_structure.base_model import *
from ml import classify
from ml import landmark_loss_compute
from ml import landmark_loss


class FaceAlignment(Model, object):
    def __init__(self, loss):
        train_dir = os.path.join(data_param['data_save_dir'], 'train')
        super(FaceAlignment, self).__init__(
            lr=face_alignment_rough_param['init_lr'],
            epochs=face_alignment_rough_param['epochs'],
            bs=face_alignment_rough_param['bs'],
            model_name=face_alignment_rough_param['model_name'],
            loss=loss,
            metrics=["accuracy"],
            steps_per_epochs=len(os.listdir(train_dir)) // (face_alignment_rough_param['bs'] * 6),
            classes=data_param['landmark_num'] * 2
        )

    def val_compute(self, imgs, labels, normalizer=None, gpu_ratio=0.5):
        # set gpu usage
        set_gpu(ratio=gpu_ratio)

        model = load_model(
            os.path.join(data_param['model_dir'], face_alignment_rough_param['model_name']),
            {'landmark_loss': self.loss})

        loss = 0.0
        count = 0
        for img, label in zip(imgs, labels):
            if normalizer:
                img = normalizer.transform(img)
            prediction = classify(model, img)
            loss += landmark_loss_compute(prediction, label)
            count += 1
            if data_param['print_debug'] and count % 100 == 0:
                logger("predicted {} imgs".format(count))
        logger("test loss is {}".format(loss / count))

    def test(self, img, mean_shape=None, normalizer=None, gpu_ratio=0.5):
        # set gpu usage
        set_gpu(ratio=gpu_ratio)

        model = load_model(os.path.join(data_param['model_dir'], face_alignment_rough_param['model_name']),
                           {'landmark_loss': landmark_loss})

        if normalizer:
            img = normalizer.transform(img)
        prediction = classify(model, img)
        print(prediction)
        print('---------------')
        if mean_shape is not None:
            prediction = np.reshape(prediction, (data_param['landmark_num'], 2)) + mean_shape
        return prediction
