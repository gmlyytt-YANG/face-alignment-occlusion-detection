#! /usr/bin/python
# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: utils.py
Author: Yang Li
Date: 2018/11/11 09:47:31
"""

import cv2
import logging
import numpy as np
import os
import scipy.io as scio
from sklearn.model_selection import train_test_split
from skimage.util import random_noise


def logger(msg):
    """Get logger format"""
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s] - %(levelname)s: %(message)s')
    logging.info(msg)


def str_or_float(obj):
    """if obj is float-like format,
        convert obj to float, else return self"""
    try:
        obj_convert = float(obj)
    except ValueError:
        obj_convert = obj
    return obj_convert


def read_pts(pts_path, landmark_num=68):
    """Read pts file and convert to matrix"""
    data = np.zeros(shape=(landmark_num, 3), dtype=np.float)
    index = 0
    for line in open(pts_path, 'r'):
        if index > 68:
            return -1
        content = line.split('\t')
        if len(content) != 4:  # \n is the end of each line
            return -1
        data[index] = [float(i) for i in content[:-1]]
        index += 1

    return data


def flip(face, pts_data):
    """Flip image"""
    face_flipped_by_x = cv2.flip(face, 1)
    landmark_ = np.asarray([(1 - x, y, _) for (x, y, _) in pts_data])

    # eye brow
    landmark_[[17, 26]] = landmark_[[26, 17]]
    landmark_[[18, 25]] = landmark_[[25, 18]]
    landmark_[[19, 24]] = landmark_[[24, 19]]
    landmark_[[20, 23]] = landmark_[[23, 20]]
    landmark_[[21, 24]] = landmark_[[24, 21]]

    # cheek
    landmark_[[0, 16]] = landmark_[[16, 0]]
    landmark_[[1, 15]] = landmark_[[15, 1]]
    landmark_[[2, 14]] = landmark_[[14, 2]]
    landmark_[[3, 13]] = landmark_[[13, 3]]
    landmark_[[4, 12]] = landmark_[[12, 4]]
    landmark_[[5, 11]] = landmark_[[11, 5]]
    landmark_[[6, 10]] = landmark_[[10, 6]]
    landmark_[[7, 9]] = landmark_[[9, 7]]

    # eyes
    landmark_[[36, 45]] = landmark_[[45, 36]]
    landmark_[[37, 44]] = landmark_[[44, 37]]
    landmark_[[38, 43]] = landmark_[[43, 38]]
    landmark_[[39, 42]] = landmark_[[42, 39]]
    landmark_[[40, 47]] = landmark_[[47, 40]]
    landmark_[[41, 46]] = landmark_[[46, 41]]

    # mouth
    landmark_[[31, 35]] = landmark_[[35, 31]]
    landmark_[[32, 34]] = landmark_[[34, 32]]
    landmark_[[48, 54]] = landmark_[[54, 48]]
    landmark_[[49, 53]] = landmark_[[53, 49]]
    landmark_[[60, 64]] = landmark_[[64, 60]]
    landmark_[[59, 55]] = landmark_[[59, 55]]
    landmark_[[50, 52]] = landmark_[[52, 50]]
    landmark_[[61, 63]] = landmark_[[63, 61]]
    landmark_[[67, 65]] = landmark_[[65, 67]]
    landmark_[[58, 56]] = landmark_[[56, 58]]

    return face_flipped_by_x, landmark_


def reproject(height, width, point, occlusion=True):
    x = width * point[0]
    y = height * point[1]
    if occlusion:
        return np.asarray([x, y, point[2]])
    else:
        return np.asarray([x, y])


def reproject_landmark(height, width, landmark, occlusion=True):
    landmark_size = 2
    if occlusion:
        landmark_size = 3
    p = np.zeros((len(landmark), landmark_size))
    for i in range(len(landmark)):
        p[i] = reproject(height, width, landmark[i], occlusion)
    return p


def project(height, width, point, occlusion=True):
    x = point[0] / width
    y = point[1] / height
    if occlusion:
        return np.asarray([x, y, point[2]])
    else:
        return np.asarray([x, y])


def project_landmark(height, width, landmark, occlusion=True):
    landmark_size = 2
    if occlusion:
        landmark_size = 3
    p = np.zeros((len(landmark), landmark_size))
    for i in range(len(landmark)):
        p[i] = project(height, width, landmark[i], occlusion)
    return p


def rotate(face, landmark, alpha):
    """Given a face with bbox and landmark, rotate with alpha
        and return rotated face with bbox, landmark (absolute position)
    """
    # convert to real size
    height = face.shape[0]
    width = face.shape[1]
    landmark_real = reproject_landmark(height, width, landmark)

    # rotate by center of image
    center = (width / 2, height / 2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    face_rotated_by_alpha = cv2.warpAffine(face, rot_mat, face.shape[:2])
    landmark_rotated = np.asarray([(rot_mat[0][0] * x + rot_mat[0][1] * y + rot_mat[0][2],
                                    rot_mat[1][0] * x + rot_mat[1][1] * y + rot_mat[1][2], _)
                                   for (x, y, _) in landmark_real])
    # landmark_num = len(landmark)
    # for landmark_index in range(landmark_num):
    #     x = int(landmark_rotated[landmark_index][0])
    #     y = int(landmark_rotated[landmark_index][1])
    #     cv2.circle(face_rotated_by_alpha, (x, y), 1, (255, 0, 0), thickness=-1)
    # show(face_rotated_by_alpha)

    # landmark projects to [0, 1]
    landmark_01 = project_landmark(height, width, landmark_rotated)
    return face_rotated_by_alpha, landmark_01


def add_postfix(name, plus_str):
    """Add postfix

    :param name: name[0] + '.' + name[1]
    :param plus_str: str to add

    :return name[0] + plus_str + '.' + name[1]
    """
    str_list = name.split('.')
    if len(str_list) != 2:
        logger("invalid name")
        return -1
    prefix_plus_name = str_list[0] + plus_str
    return prefix_plus_name + '.' + str_list[1]


def data_append(faces, landmarks, occlusions, names,
                face, landmark, occlusion, name):
    faces.append(face)
    landmarks.append(landmark)
    occlusions.append(occlusion)
    names.append(name)


def data_aug(face, pts_data, img_size, color, name):
    """Data augment
    :param face:
    :param pts_data:
    :param img_size:
    :param color:
    :param name: name of the face
    :return: faces, landmarks, names
    """
    assert face is not None
    faces = []
    landmarks = []
    occlusions = []
    names = []
    alpha = 5  # rotate degree

    # flip1
    face_flipped, landmark_flipped = flip(face, pts_data)
    occlusion_flipped = landmark_flipped[:, 2]
    face_flipped = cv2.resize(face_flipped, (img_size, img_size))
    data_append(faces, landmarks, occlusions, names,
                face_flipped, np.clip(landmark_flipped[:, :-1], 0, 1),
                occlusion_flipped, add_postfix(name, "_flip"))

    # rotation1
    face_rotated_by_alpha, landmark_rotated = rotate(face, pts_data, alpha)
    landmark_rotated[:, :2] = np.clip(landmark_rotated[:, :2], 0, 1)  # exception dealing
    occlusion_rotated = landmark_rotated[:, 2]
    face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (img_size, img_size))
    data_append(faces, landmarks, occlusions, names,
                face_rotated_by_alpha, landmark_rotated[:, :-1],
                occlusion_rotated, add_postfix(name, "_rotation_{}".format(alpha)))

    # flip with rotation1
    face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
    occlusion_flipped = landmark_flipped[:, 2]
    face_flipped = cv2.resize(face_flipped, (img_size, img_size))
    data_append(faces, landmarks, occlusions, names,
                face_flipped, landmark_flipped[:, :-1],
                occlusion_flipped, add_postfix(name, "_rotation_{}_flip".format(alpha)))

    # rotation2
    face_rotated_by_alpha, landmark_rotated = rotate(face, pts_data, -alpha)
    landmark_rotated[:, :2] = np.clip(landmark_rotated[:, :2], 0, 1)
    occlusion_rotated = landmark_rotated[:, 2]
    face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (img_size, img_size))
    data_append(faces, landmarks, occlusions, names,
                face_rotated_by_alpha, landmark_rotated[:, :-1],
                occlusion_rotated, add_postfix(name, "_rotation_-{}".format(alpha)))

    # flip with rotation2
    face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
    occlusion_flipped = landmark_flipped[:, 2]
    face_flipped = cv2.resize(face_flipped, (img_size, img_size))
    data_append(faces, landmarks, occlusions, names,
                face_flipped, landmark_rotated[:, :-1],
                occlusion_flipped, add_postfix(name, "_rotation_-{}_flip".format(alpha)))

    # origin
    face = cv2.resize(face, (img_size, img_size))
    data_append(faces, landmarks, occlusions, names,
                face, pts_data[:, :2], pts_data[:, 2], name)

    return faces, landmarks, occlusions, names


def point_in_bbox(pt, bbox):
    """Judge the pt is in the bbox"""
    assert len(bbox) == 4
    assert len(pt) == 2
    x = pt[0]
    y = pt[1]
    if bbox[0] <= x <= bbox[1] and bbox[2] <= y <= bbox[3]:
        return True
    return False


def show(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)


def range_search(x, y, patch_size, face_size, maintain_radius=True):
    left = x - patch_size / 2 if x - patch_size / 2 >= 0 else 0
    top = y - patch_size / 2 if y - patch_size / 2 >= 0 else 0

    if maintain_radius:
        # right out of bound
        if left + patch_size >= face_size:
            left = face_size - patch_size
            # right bottom out of bound
            if top + patch_size >= face_size:
                top = face_size - patch_size

        # left bottom out of bound
        elif top + patch_size >= face_size:
            top = face_size - patch_size

        right = left + patch_size
        bottom = top + patch_size
    else:
        right = x + patch_size / 2 if x + patch_size / 2 <= face_size else face_size
        bottom = y + patch_size / 2 if y + patch_size / 2 <= face_size else face_size

    return int(left), int(right), int(top), int(bottom)


def get_patch(imgs, landmarks, occlusions, patch_size, patches, labels, landmark_num):
    """Get patch of each landmark
    :param imgs:
    :param landmarks:
    :param occlusions:
    :param patch_size:
    :param patches: patch container
    :param labels: label container
    :return:
    """
    data_size = len(imgs)
    for index in range(data_size):
        face = imgs[index]
        face_size = face.shape[0]
        landmark = landmarks[index]
        occlusion = occlusions[index]
        landmark = reproject_landmark(face_size, face_size, landmark, occlusion=False)
        for landmark_index in range(landmark_num):
            [x_center, y_center] = landmark[landmark_index]
            left, right, top, bottom = range_search(x_center, y_center, patch_size, face_size)
            face_part = face[int(top): int(bottom), int(left): int(right)]
            patches[landmark_index].append(face_part)
            labels[landmark_index].append(occlusion[landmark_index])


def dataset_split(x, y, test_size=0.3, random_state=0):
    """Dataset Splitting using sklearn

    :param x: image dataset
    :param y: label
    :param test_size:
    :param random_state:
    :return:
    """
    logger("len x is {}, len y is {}".format(len(x), len(y)))
    assert len(x) == len(y)

    # data splitting
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True,
                                                        test_size=test_size,
                                                        random_state=random_state)
    train_data = {'data': x_train, 'label': y_train}
    validation_data = {'data': x_test, 'label': y_test}
    return train_data, validation_data


def heat_map_dist(point, matrix):
    sigma = 0.05
    D = np.sqrt(np.sum((point - matrix) ** 2))
    M = np.exp(np.multiply(D ** 2, - 2 * sigma ** 2))
    return M


def color(landmark_elem, face_size, heat_map_mask, radius):
    x = landmark_elem[0]
    y = landmark_elem[1]
    left, right, top, bottom = range_search(x, y, radius * 2, face_size, maintain_radius=False)
    for x_elem in range(left, right):
        for y_elem in range(top, bottom):
            heat = heat_map_dist([x_elem, y_elem], landmark_elem)
            if heat > heat_map_mask[y_elem][x_elem]:
                heat_map_mask[y_elem][x_elem] = heat


def heat_map_compute(param):
    """Heat map compute

    :param param: dict{"face", "landmark", "landmark_01"}
    face: face image
    landmark: landmark points set
    landmark_01: whether landmark is normalized
    radius:
    """
    face = param['face']
    landmark = param['landmark']
    landmark_is_01 = param['landmark_01']
    radius = param['radius']
    face_size = face.shape[:2]
    heat_map_mask = np.zeros_like(face[:, :, 0], dtype=np.float)
    if landmark_is_01:
        landmark = np.multiply(landmark, np.array([face_size[1], face_size[0]]))
    landmark = landmark.astype(int)
    # draw_landmark(face, landmark)
    for landmark_elem in landmark:
        color(landmark_elem, face_size[0], heat_map_mask, radius)
    heat_map_mask = heat_map_mask[:, :, np.newaxis]
    heat_map_mask = heat_map_mask.repeat([3], axis=2)
    heat_map = np.multiply(face, heat_map_mask)
    # show(heat_map_mask)
    # show(heat_map)
    return heat_map


def get_face(img, bbox, need_to_convert_to_int=False):
    if need_to_convert_to_int:
        bbox = [int(_) for _ in bbox]
    face = img[bbox[2]:bbox[3], bbox[0]: bbox[1]]
    return face


def load_basic_info(mat_file, img_root=None):
    """Load basic info

    :param mat_file: mat file name including nameList, bbox
    :param img_root: img_root dir

    :return img_paths: abs paths of images
    :return bboxes:
    """
    data = scio.loadmat(os.path.join(img_root, mat_file))
    img_paths = data['nameList']
    img_paths = [os.path.join(img_root, i[0][0]) for i in img_paths]
    bboxes = data['bbox']
    return img_paths, bboxes


def load_imgs(img_root, mat_file_name, total=True, chosed="random"):
    """Load images

    :param img_root: img root dir
    :param mat_file_name:
    :param total: bool obj, control to return whole dataset or just part
    :param chosed: whether to choose specific indices of dataset or just random

    :return chosed objs
    """
    img_paths, bboxes = load_basic_info(mat_file_name, img_root)
    if not total:
        if isinstance(chosed, list):
            faces = []
            for index in chosed:
                print(img_paths[index])
                img = cv2.imread(img_paths[index])
                faces.append(get_face(img, bboxes[index], need_to_convert_to_int=True))
            return faces
        elif chosed == "random":
            length = len(img_paths)
            index = np.random.randint(0, length)
            img = img_paths[index]
            face = get_face(img, bboxes[index], need_to_convert_to_int=True)
            return face

    return (get_face(cv2.imread(img_paths[index]), bboxes[index], need_to_convert_to_int=True)
            for index in range(len(img_paths)))


def draw_landmark(img, landmarks):
    for i in range(len(landmarks)):
        for idx, point in enumerate(landmarks):
            pos = (int(point[0]), int(point[1]))
            cv2.circle(img, pos, 5, color=(0, 255, 0))
    cv2.namedWindow("img", 2)
    cv2.imshow("img", img)
    cv2.waitKey(0)


def gaussian_noise(img, mode='gaussian'):
    """Add gaussian noise to images """
    # show(img)
    # for index in range(noise_num):
    #     x = np.random.randint(0, img.shape[1])
    #     y = np.random.randint(0, img.shape[0])
    #     noised_img[y][x] = 255
    noised_img = random_noise(img, mode=mode, clip=True)
    # show(noised_img)
    return noised_img


def binary(num, threshold):
    num = 1 if num > threshold else 0
    return num


def occlu_ratio_compute(labels):
    count = 0
    for index in range(len(labels)):
        if np.sum(labels[index]) > 0:
            count += 1
    return float(count) / len(labels)


def accuracy_compute(labels, predictions):
    count = 0
    for index in range(len(labels)):
        if (labels[index] == predictions[index]).all():
            count += 1
    result = float(count) / len(labels)
    return result


def recall_compute(labels, predictions, mode="occlu"):
    count = 0
    good_count = 0
    if mode == "occlu":
        flag = 2
    else:
        flag = 0
    for index in range(len(labels)):
        for index_inner in range(len(labels[index])):
            if labels[index][index_inner] + \
                    predictions[index][index_inner] == flag:
                good_count += 1
            if labels[index][index_inner] == int(flag / 2):
                count += 1
    return float(good_count) / count
