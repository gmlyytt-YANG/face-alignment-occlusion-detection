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
from sklearn.model_selection import train_test_split


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


def data_aug(face, pts_data, img_size, color):
    """Data augment
    :param face:
    :param pts_data:
    :param bbox:
    :param img_size:
    :param color:
    :return: faces, landmarks
    """
    assert face is not None
    faces = []
    landmarks = []
    occlusions = []
    alpha = 5  # rotate degree

    # flip1
    face_flipped, landmark_flipped = flip(face, pts_data)
    occlusion_flipped = landmark_flipped[:, 2]
    face_flipped = cv2.resize(face_flipped, (img_size, img_size))
    faces.append(face_flipped)
    landmarks.append(np.clip(landmark_flipped[:, :-1], 0, 1))
    occlusions.append(occlusion_flipped)

    # rotation1
    face_rotated_by_alpha, landmark_rotated = rotate(face, pts_data, alpha)
    landmark_rotated[:, :2] = np.clip(landmark_rotated[:, :2], 0, 1)  # exception dealing
    occlusion_rotated = landmark_rotated[:, 2]
    face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (img_size, img_size))
    faces.append(face_rotated_by_alpha)
    landmarks.append(landmark_rotated[:, :-1])
    occlusions.append(occlusion_rotated)

    # flip with rotation1
    face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
    occlusion_flipped = landmark_flipped[:, 2]
    face_flipped = cv2.resize(face_flipped, (img_size, img_size))
    faces.append(face_flipped)
    landmarks.append(landmark_flipped[:, :-1])
    occlusions.append(occlusion_flipped)

    # rotation2
    face_rotated_by_alpha, landmark_rotated = rotate(face, pts_data, -alpha)
    landmark_rotated[:, :2] = np.clip(landmark_rotated[:, :2], 0, 1)
    occlusion_rotated = landmark_rotated[:, 2]
    face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (img_size, img_size))
    faces.append(face_rotated_by_alpha)
    landmarks.append(landmark_rotated[:, :-1])
    occlusions.append(occlusion_rotated)

    # flip with rotation2
    face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
    occlusion_flipped = landmark_flipped[:, 2]
    face_flipped = cv2.resize(face_flipped, (img_size, img_size))
    faces.append(face_flipped)
    landmarks.append(landmark_flipped[:, :-1])
    occlusions.append(occlusion_flipped)

    # origin
    face = cv2.resize(face, (img_size, img_size))
    faces.append(face)
    landmarks.append(pts_data[:, :2])
    occlusions.append(pts_data[:, 2])

    return faces, landmarks, occlusions


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


def range_search(x, y, patch_size, face_size):
    left = x - patch_size / 2 if x - patch_size / 2 >= 0 else 0
    top = y - patch_size / 2 if y - patch_size / 2 >= 0 else 0

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

    return left, right, top, bottom


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
    D = np.min(np.sqrt(np.sum((point - matrix) ** 2, axis=1)))
    M = np.exp(np.multiply(D ** 2, - 2 * sigma ** 2))
    return M


def heat_map_compute(param):
    """Heat map compute

    :param param: dict{"face", "landmark", "landmark_01"}
    face: face image
    landmark: landmark points set
    landmark_01: whether landmark is normalized
    """
    face = param['face']
    landmark = param['landmark']
    landmark_is_01 = param['landmark_01']
    face_size = face.shape[:2]
    heat_map_mask = np.zeros_like(face[:, :, 0], dtype=np.float)
    if landmark_is_01:
        landmark = np.multiply(landmark, np.array([face_size[1], face_size[0]]))
    for x in range(face_size[1]):
        for y in range(face_size[0]):
            heat_map_mask[y][x] = heat_map_dist([x, y], landmark)
    heat_map_mask = heat_map_mask[:, :, np.newaxis]
    heat_map_mask = heat_map_mask.repeat([3], axis=2)
    heat_map = np.multiply(face, heat_map_mask)
    # show(heat_map)
    return heat_map


def generate_batch_data_random(x, y, batch_size):
    length = len(y)
    loop_count = length // batch_size
    while True:
        i = np.random.randint(0, loop_count)
        yield x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size]
