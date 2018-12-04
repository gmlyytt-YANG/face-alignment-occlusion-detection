#! /usr/bin/python

import cv2
import tensorflow as tf

from utils import *
import pickle

# data_dir = './data/train'
# f = open('./model/mean_shape.pkl', 'rb')
# mean_shape = pickle.load(f)
#
# img_name_list, label_name_list = get_filenames(data_dir, listext=["*_face.png", "*_face.jpg"], label_ext=".pts")
#
# for img_name, label_name in zip(img_name_list, label_name_list):
#     img = cv2.imread(img_name)
#     landmark = np.genfromtxt(label_name)
#     # draw_landmark(img, landmark)
#     draw_landmark(img, mean_shape)

x = tf.constant([[1, 2, 3], [4, 5, 6]])

tf.reduce_max(x, 0)  # [4, 5, 6]
y = tf.reduce_max(x, 1)  # [3, 6]
tf.reduce_max(x, 1, keep_dims=True)  # [[3], [6]]
tf.reduce_max(x, [0, 1])  # 6

with tf.Session() as sess:
    print(sess.run(y))

