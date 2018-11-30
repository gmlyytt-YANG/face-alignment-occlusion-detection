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

from config.init_param import occlu_param, data_param
from model_structure.smaller_vggnet import SmallerVGGNet
from model_structure.new_vgg16net import Vgg16Net
from ml import *
from prepare.occlu_data_gen import train_data_feed, validation_data_feed


class Model(object):
    def __init__(self):
        self.print_debug = data_param['print_debug']

    def train(self, param):
        pass


class OcclusionDetection(object):
    def __init__(self):
        self.print_debug = data_param['print_debug']

    def train(self, model_type="vgg16", gpu_ratio=0.5):
        # set gpu usage
        set_gpu(ratio=gpu_ratio)

        # load config
        data_save_dir = data_param['data_save_dir']
        model_dir = data_param['model_dir']
        img_size = data_param['img_size']
        channel = data_param['channel']
        classes = data_param['landmark_num']
        model_name = occlu_param['model_name']
        epochs = occlu_param['epochs']
        bs = occlu_param['bs']
        lr = occlu_param['init_lr']

        # loading data
        logger("loading data")
        train_dir = os.path.join(data_save_dir, "train")
        validation_dir = os.path.join(data_save_dir, "val")
        validation_data, validation_labels = \
            validation_data_feed(validation_dir, self.print_debug)

        # build model
        logger("building model")
        opt = Adam(lr=lr, decay=lr / epochs)
        model_structure = Vgg16Net if model_type == "vgg16" else SmallerVGGNet
        model = model_structure.build(width=img_size, height=img_size,
                                      depth=channel, classes=classes,
                                      final_act="sigmoid")

        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

        # train model
        logger("training")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        checkpoint = ModelCheckpoint(filepath=os.path.join(model_dir, model_name))
        early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=2)
        callback_list = [checkpoint, early_stopping]
        model.fit_generator(
            train_data_feed(bs, train_dir),
            validation_data=(validation_data, validation_labels),
            steps_per_epoch=len(os.listdir(train_dir)) // (bs * 6),
            epochs=epochs, verbose=1, callbacks=callback_list)

        K.clear_session()

    @staticmethod
    def classify(model, img, need_to_normalize=False):
        img = cv2.resize(img, (data_param['img_size'], data_param['img_size']))
        if need_to_normalize:
            img = img.astype("float") / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        prob = model.predict(img)[0]
        return prob

    def validation_benchmark(self):
        # set gpu usage
        set_gpu(ratio=0.5)

        # load model
        model = load_model(
            os.path.join(data_param['model_dir'], occlu_param['model_name']))

        # load data
        validation_dir = os.path.join(data_param['data_save_dir'], "val")
        validation_data, validation_labels = \
            validation_data_feed(validation_dir, self.print_debug)

        # forward
        predict_labels = []
        length = len(validation_data)
        for index in range(length):
            prediction = [binary(_, threshold=0.5)
                          for _ in self.classify(model, validation_data[index])]
            predict_labels.append(prediction)
            if self.print_debug and (index + 1) % 500 == 0:
                logger("predicted {} imgs".format(index + 1))
        K.clear_session()

        # compute 
        logger("the result of prediction of validation is as follow:")
        occlu_ratio = occlu_ratio_compute(validation_labels)
        accuracy = accuracy_compute(validation_labels, predict_labels)
        occlu_recall = recall_compute(validation_labels, predict_labels, mode="occlu")
        clear_recall = recall_compute(validation_labels, predict_labels, mode="clear")
        print("occlu_ratio is {}".format(occlu_ratio))
        print("accuracy is {}".format(accuracy))
        print("occlu_recall is {}".format(occlu_recall))
        print("clear_recall is {}".format(clear_recall))
