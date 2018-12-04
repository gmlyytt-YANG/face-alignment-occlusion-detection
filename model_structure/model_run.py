#! /usr/bin/python
# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: model_run.py
Author: Yang Li
Date: 2018/11/10 17:43:31
Description: Derived Model
"""
from keras.models import load_model
from model_structure.base_model import *
from ml import *


class OcclusionDetection(Model, object):
    def __init__(self):
        train_dir, val_dir = self._load_data()
        super(OcclusionDetection, self).__init__(
            lr=occlu_param['init_lr'],
            epochs=occlu_param['epochs'],
            bs=occlu_param['bs'],
            final_act="sigmoid",
            loss=occlu_param['loss'],
            metrics=["accuracy"],
            steps_per_epochs=len(os.listdir(train_dir)) // (occlu_param['bs'] * 6),
            train_dir=train_dir,
            val_dir=val_dir,
            classmethod=data_param['landmark_num']
        )

    def val_compute(self, val_load, ext_lists, label_ext, normalizer=None, gpu_ratio=0.5):
        # set gpu usage
        set_gpu(ratio=gpu_ratio)

        # load model
        model = load_model(
            os.path.join(data_param['model_dir'], occlu_param['model_name']))

        # load data
        val_data, val_labels = val_load(data_dir=self.val_dir,
                                        ext_lists=ext_lists,
                                        label_ext=label_ext,
                                        normalizer=normalizer,
                                        print_debug=self.print_debug)

        # forward
        predict_labels = []
        length = len(val_data)
        for index in range(length):
            prediction = [binary(_, threshold=0.5)
                          for _ in self.classify(model, val_data[index])]
            predict_labels.append(prediction)
            if self.print_debug and (index + 1) % 500 == 0:
                logger("predicted {} imgs".format(index + 1))
        K.clear_session()

        # compute
        logger("the result of prediction of validation is as follow:")
        occlu_ratio = occlu_ratio_compute(val_labels)
        accuracy = accuracy_compute(val_labels, predict_labels)
        occlu_recall = recall_compute(val_labels, predict_labels, mode="occlu")
        clear_recall = recall_compute(val_labels, predict_labels, mode="clear")
        print("occlu_ratio is {}".format(occlu_ratio))
        print("accuracy is {}".format(accuracy))
        print("occlu_recall is {}".format(occlu_recall))
        print("clear_recall is {}".format(clear_recall))

    def test(self, model, img, landmark, is_heat_map=False, binary_output=False):
        normalizer = np.load(os.path.join(data_param['normalizer_dir'], "normalizer.npz"))
        img = cv2.resize(img, (data_param['img_height'], data_param['img_width']))
        net_input = img
        if is_heat_map:
            net_input = heat_map_compute(img, landmark,
                                         landmark_is_01=False,
                                         img_color=True,
                                         radius=occlu_param['radius'])
        if binary_output:
            return [binary(_, threshold=0.5)
                    for _ in self.classify(model, net_input, normalizer)]
        return self.classify(model, net_input)


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
