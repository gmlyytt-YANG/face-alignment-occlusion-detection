#! /usr/bin/python
# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: gpu_usage.py
Author: Yang Li
Date: 2018/11/26 16:33:31
Description: gpu limitation
"""

import os


def set_gpu(ratio=0, target='memory'):
    command1 = "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free | awk '{print $3}'"
    command2 = "nvidia-smi -q | grep Gpu | awk '{print $3}'"
    memory = list(map(int, os.popen(command1).readlines()))
    gpu = list(map(int, os.popen(command2).readlines()))
    if memory and gpu:  # 如果没有显卡，memory，gpu均为[]
        if target == 'memory':
            num = (1, 0)[memory[0] > memory[1]]
        else:
            num = (0, 1)[gpu[0] > gpu[1]]
        print('>>> Free Memory       : GPU0 %6d MiB | GPU1 %6d MiB' % (memory[0], memory[1]))
        print('>>> Volatile GPU-Util : GPU0 %6d %%   | GPU1 %6d %% ' % (gpu[0], gpu[1]))
        print('>>> Using GPU%d' % num)
        import tensorflow as tf
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = str(num)  # 选择GPU
        if ratio == 0:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = ratio
        sess = tf.Session(config=config)
        from keras import backend as K
        K.set_session(sess)
    else:
        print('>>> Could not find the GPU')
