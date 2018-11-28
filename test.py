import os

data_dir = "/home/kb/Documents/yl/0_DATASET/origin_img/LFPW68/testset"

for path in os.listdir(data_dir):
    file = os.path.join(data_dir, path)
    if file.split(".")[-1] == "pts_occlu":
        new_file = os.path.splitext(file)[0] + ".opts"
        f_new = open(new_file, 'w')
        f = open(file, 'r')
        for line in f.readlines():
            line = line.strip()
            line = line.replace("\t", " ")
            f_new.write(line + "\n")
        f.close()
        f_new.close()

from utils import draw_landmark
import cv2
import numpy as np

img = cv2.imread(
    "/home/kb/Documents/yl/face-alignment-occlusion-detection/data/train/HELEN68_trainset_100843687_1_rotation_10_face.jpg")
face_size = img.shape[:2]
landmark = np.genfromtxt(
    "/home/kb/Documents/yl/face-alignment-occlusion-detection/data/train/HELEN68_trainset_100843687_1_rotation_10_face.pts",
    dtype=float, delimiter=" ")
landmark = np.multiply(landmark, np.array([face_size[1], face_size[0]]))
draw_landmark(img, landmark)
