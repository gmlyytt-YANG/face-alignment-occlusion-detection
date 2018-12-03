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
Description: Data Augment
"""

from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

from config.init_param import *
from ml import *
from prepare.data_gen import *


class Model(object):
    def __init__(self, lr, epochs, bs,  train_dir, val_dir,
                 loss, metrics, steps_per_epochs, final_act=None):
        self.print_debug = data_param['print_debug']
        self.data_save_dir = data_param['data_save_dir']
        self.model_dir = data_param['model_dir']
        self.width = data_param['img_width']
        self.height = data_param['img_height']
        self.classes = data_param['landmark_num']
        self.channel = occlu_param['channel']
        self.model_name = occlu_param['model_name']
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.lr = lr
        self.epochs = epochs
        self.bs = bs
        self.final_act = final_act
        self.loss = loss
        self.metrics = metrics
        self.steps_per_epoch = steps_per_epochs

    @staticmethod
    def _load_data():
        train_dir = os.path.join(data_param['data_save_dir'], "train")
        validation_dir = os.path.join(data_param['data_save_dir'], "val")
        return train_dir, validation_dir

    def train(self, model_structure, train_load, val_load,
              ext_lists, label_ext, gpu_ratio=0.5):
        # set gpu usage
        set_gpu(ratio=gpu_ratio)

        # load data
        logger("loading data")
        val_data, val_labels = val_load(data_dir=self.val_dir,
                                        ext_lists=ext_lists,
                                        label_ext=label_ext,
                                        print_debug=self.print_debug)

        # build model
        logger("building model")
        opt = Adam(lr=self.lr, decay=self.lr / self.epochs)
        model = model_structure.build(width=self.width, height=self.height,
                                      depth=self.channel, classes=self.classes,
                                      final_act=self.final_act)
        model.compile(loss=self.loss, optimizer=opt, metrics=self.metrics)

        logger("training")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        checkpoint = ModelCheckpoint(filepath=os.path.join(self.model_dir, self.model_name))
        early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=2)
        callback_list = [checkpoint, early_stopping]
        model.fit_generator(
            train_load(batch_size=self.bs, data_dir=self.train_dir,
                       ext_lists=ext_lists, label_ext=label_ext),
            validation_data=(val_data, val_labels),
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs, verbose=1, callbacks=callback_list)

        K.clear_session()

    @staticmethod
    def classify(model, img, need_to_normalize=False):
        img = cv2.resize(img, (data_param['img_height'], data_param['img_width']))
        if need_to_normalize:
            img = img.astype("float") / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        prob = model.predict(img)[0]
        return prob


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
            val_dir=val_dir
        )

    def validation_benchmark(self):
        # set gpu usage
        set_gpu(ratio=0.5)

        # load model
        model = load_model(
            os.path.join(data_param['model_dir'], occlu_param['model_name']))

        # load data
        val_data, val_labels = validation_data_feed(data_dir=self.val_dir,
                                                    ext_lists=["*_heatmap.png", "*_heatmap.jpg"],
                                                    label_ext=".opts",
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


class FaceAlignmentRough(Model, object):
    def __init__(self):
        train_dir, val_dir = self._load_data()
        super(FaceAlignmentRough, self).__init__(
            lr=face_alignment_rough_param['init_lr'],
            epochs=face_alignment_rough_param['epochs'],
            bs=face_alignment_rough_param['bs'],
            loss="mean_squared_error",
            metrics=["accuracy"],
            steps_per_epochs=len(os.listdir(train_dir)) // (occlu_param['bs'] * 6),
            train_dir=train_dir,
            val_dir=val_dir
        )