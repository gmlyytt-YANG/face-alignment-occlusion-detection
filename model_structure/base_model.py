#! /usr/bin/python
# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: base_model.py
Author: Yang Li
Date: 2018/11/10 17:43:31
Description: Base Model
"""

import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

from config.init_param import *
from prepare.data_gen import *


class Model(object):
    def __init__(self, lr, epochs, bs, train_dir, val_dir,
                 loss, metrics, steps_per_epochs, classes, final_act=None):
        self.print_debug = data_param['print_debug']
        self.data_save_dir = data_param['data_save_dir']
        self.model_dir = data_param['model_dir']
        self.width = data_param['img_width']
        self.height = data_param['img_height']
        self.classes = classes
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
              ext_lists, label_ext, mean_data=None, gpu_ratio=0.5):
        # set gpu usage
        set_gpu(ratio=gpu_ratio)

        # load data
        logger("loading data")
        val_data, val_labels = val_load(data_dir=self.val_dir,
                                        ext_lists=ext_lists,
                                        label_ext=label_ext,
                                        mean_data=mean_data,
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
        H = model.fit_generator(
            train_load(batch_size=self.bs, data_dir=self.train_dir,
                       ext_lists=ext_lists, label_ext=label_ext, mean_data=mean_data),
            validation_data=(val_data, val_labels),
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs, verbose=1, callbacks=callback_list)
       
        # plt.style.use("ggplot")
        # plt.figure()
        # plt.plot(np.arange(0, self.epochs), H.history["loss"], label="train_loss")
        # plt.plot(np.arange(0, self.epochs), H.history["val_loss"], label="val_loss")
        # plt.plot(np.arange(0, self.epochs), H.history["acc"], label="train_acc")
        # plt.plot(np.arange(0, self.epochs), H.history["val_acc"], label="val_acc")
        # plt.title("Training Loss and Accuracy")
        # plt.xlabel("Epoch #")
        # plt.ylabel("Loss/Accuracy")
        # plt.legend(loc="upper right")
        # plt.savefig("Loss_Accuracy_{:d}e.jpg".format(epochs))

        K.clear_session()

    @staticmethod
    def classify(model, img, mean_data=None):
        if mean_data is not None:
            img = mean_data.transform(img)
        if img.shape[:2] != [data_param['img_height'], data_param['img_width']]:
            img = cv2.resize(img, (data_param['img_height'], data_param['img_width']))
        img = np.expand_dims(img_to_array(img), axis=0)
        return model.predict(img)[0]
