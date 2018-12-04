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
import glob
import numpy as np
import os
import scipy.io as scio


def logger(msg):
    """Get logger format"""
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s] - %(levelname)s: %(message)s')
    logging.info(msg)


def remove_content(path):
    if os.path.isfile(path):
        os.remove(path)
    else:
        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_gpu(ratio=0):
    command1 = "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free | awk '{print $3}'"
    command2 = "nvidia-smi -q | grep Gpu | awk '{print $3}'"
    memory = list(map(int, os.popen(command1).readlines()))
    gpu = list(map(int, os.popen(command2).readlines()))
    if memory and gpu:
        import tensorflow as tf
        config = tf.ConfigProto()
        if ratio == 0:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = ratio
        sess = tf.Session(config=config)
        from keras import backend as K
        K.set_session(sess)


def str_or_float(obj):
    """if obj is float-like format,
        convert obj to float, else return self"""
    try:
        obj_convert = float(obj)
    except ValueError:
        obj_convert = obj
    return obj_convert


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


def data_aug(face, landmark, name):
    """Data augment
    :param face:
    :param landmark:
    :param name: name of the face
    :return: faces, landmarks, names
    """
    assert face is not None
    faces = []
    landmarks = []
    occlusions = []
    names = []
    alpha = 5  # rotate degree

    # 1. flip
    face_flipped, landmark_flipped = flip(face, landmark)
    occlusion_flipped = landmark_flipped[:, 2].astype(int)
    data_append(faces, landmarks, occlusions, names,
                face_flipped, np.clip(landmark_flipped[:, :-1], 0, 1), occlusion_flipped,
                add_postfix(name, "_flip").replace("/", "_"))

    # 2. rotation_5
    face_rotated_by_alpha, landmark_rotated = rotate(face, landmark, alpha)
    landmark_rotated[:, :2] = np.clip(landmark_rotated[:, :2], 0, 1)  # exception dealing
    occlusion_rotated = landmark_rotated[:, 2].astype(int)
    data_append(faces, landmarks, occlusions, names,
                face_rotated_by_alpha, landmark_rotated[:, :-1], occlusion_rotated,
                add_postfix(name, "_rotation_{}".format(alpha)))

    # 3. rotation_5_flip
    face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
    occlusion_flipped = landmark_flipped[:, 2].astype(int)
    data_append(faces, landmarks, occlusions, names,
                face_flipped, landmark_flipped[:, :-1], occlusion_flipped,
                add_postfix(name, "_rotation_{}_flip".format(alpha)))

    # 4. rotation_10
    face_rotated_by_alpha, landmark_rotated = rotate(face, landmark, 2 * alpha)
    landmark_rotated[:, :2] = np.clip(landmark_rotated[:, :2], 0, 1)  # exception dealing
    occlusion_rotated = landmark_rotated[:, 2].astype(int)
    data_append(faces, landmarks, occlusions, names,
                face_rotated_by_alpha, landmark_rotated[:, :-1], occlusion_rotated,
                add_postfix(name, "_rotation_{}".format(2 * alpha)))

    # 5. rotation_-5
    face_rotated_by_alpha, landmark_rotated = rotate(face, landmark, -alpha)
    landmark_rotated[:, :2] = np.clip(landmark_rotated[:, :2], 0, 1)
    occlusion_rotated = landmark_rotated[:, 2].astype(int)
    data_append(faces, landmarks, occlusions, names,
                face_rotated_by_alpha, landmark_rotated[:, :-1], occlusion_rotated,
                add_postfix(name, "_rotation_-{}".format(alpha)))

    # 6. rotation_-5_flip
    face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
    occlusion_flipped = landmark_flipped[:, 2].astype(int)
    data_append(faces, landmarks, occlusions, names,
                face_flipped, landmark_flipped[:, :-1], occlusion_flipped,
                add_postfix(name, "_rotation_-{}_flip".format(alpha)))

    # 7. rotation_-10
    face_rotated_by_alpha, landmark_rotated = rotate(face, landmark, -2 * alpha)
    landmark_rotated[:, :2] = np.clip(landmark_rotated[:, :2], 0, 1)
    occlusion_rotated = landmark_rotated[:, 2].astype(int)
    data_append(faces, landmarks, occlusions, names,
                face_rotated_by_alpha, landmark_rotated[:, :-1], occlusion_rotated,
                add_postfix(name, "_rotation_-{}".format(2 * alpha)))

    # 8. origin
    data_append(faces, landmarks, occlusions, names,
                face, landmark[:, :2], landmark[:, 2].astype(int), name)

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
    """Generate square box around [x, y] in 2*face_size square

    :param: x, y
    :param: patch_size: side length of square
    :param: face_size:
    :param: maintain_raidus: control area of square, if True, the area of square is
                             face_size^2
    """
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


def heat_map_dist(point, matrix):
    """Heat_map distance compute method

    :param: point: one point.
    :param: matrix: points set.
    """
    sigma = 0.05
    D = np.sqrt(np.sum((point - matrix) ** 2))
    M = np.exp(np.multiply(D ** 2, - 2 * sigma ** 2))
    return M


def color(landmark_elem, face_size, heat_map_mask, radius):
    """Color the img according to landmark_elem

    :param: landmark_elem: one landmark coordinate.
    :param: face_size:
    :param: heat_map_mask: result container.
    :param: radius: color range.
    """
    x = landmark_elem[0]
    y = landmark_elem[1]
    left, right, top, bottom = range_search(x, y, radius * 2, face_size, maintain_radius=False)
    for x_elem in range(left, right):
        for y_elem in range(top, bottom):
            heat = heat_map_dist([x_elem, y_elem], landmark_elem)
            if heat > heat_map_mask[y_elem][x_elem]:
                heat_map_mask[y_elem][x_elem] = heat


def normalize_data(landmark, bbox=None, occlu_include=True):
    """Normalize landmark

    :param: landmark:
    :param: bbox:
    :param: occlu_include: bool like obj, control whether landmark has col2(0 starting index)
    """
    if bbox is None:
        min_x, min_y = np.min(landmark[:, :2], axis=0)
        w, h = np.ptp(landmark[:, :2], axis=0)
    else:
        min_x, min_y = bbox[0], bbox[2]
        w, h = bbox[1] - bbox[0], bbox[3] - bbox[2]
    normalized_landmark = (landmark[:, :2] - [min_x, min_y]) / [w, h]
    if occlu_include:
        return np.hstack((normalized_landmark, np.expand_dims(landmark[:, 2], axis=1)))
    return normalized_landmark


def heat_map_compute(face, landmark, landmark_is_01, img_color, radius):
    """Heat map compute

    :param face: face image
    :param landmark: landmark points set
    :param landmark_is_01: whether landmark is normalized
    :param radius:
    """
    face_size = face.shape[:2]
    if img_color:
        heat_map_mask = np.zeros_like(face[:, :, 0], dtype=np.float)
    else:
        heat_map_mask = np.zeros_like(face, dtype=np.float)
    if landmark_is_01:
        landmark = np.multiply(landmark, np.array([face_size[1], face_size[0]]))
    landmark = landmark.astype(int)
    # draw_landmark(face, landmark)
    for landmark_elem in landmark:
        color(landmark_elem, face_size[0], heat_map_mask, radius)
    if img_color:
        heat_map_mask = heat_map_mask[:, :, np.newaxis].repeat([3], axis=2)
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


def load_rough_imgs_labels_core(img_path, bbox, img_size, normalizer=None):
    img = cv2.imread(img_path)
    face = cv2.resize(get_face(img, bbox, need_to_convert_to_int=True),
                      (img_size, img_size))
    if normalizer:
        face = normalizer.transform(face)
    label_path = os.path.splitext(img_path)[0] + ".pts"

    label = np.multiply(np.clip(
        normalize_data(np.genfromtxt(label_path, skip_header=3, skip_footer=1),
                       bbox, occlu_include=False), 0, 1), img_size)
    return face, label


def load_rough_imgs_labels(img_root, mat_file_name, img_size,
                           normalizer=None, chosen="random"):
    """Load rough imgs and labels without normalization.

    :param img_root: img root dir
    :param mat_file_name:
    :param img_size:
    :param normalizer:
    :param chosen: whether to choose specific indices of dataset or just random

    :return chosen objs
    """
    img_paths, bboxes = load_basic_info(mat_file_name, img_root)
    if chosen == "random":
        length = len(img_paths)
        index = np.random.randint(0, length)
        return load_rough_imgs_labels_core(img_path=img_paths[index],
                                           bbox=bboxes[index],
                                           img_size=img_size,
                                           normalizer=normalizer)
    else:
        faces = []
        labels = []
        for index in chosen:
            face, label = load_rough_imgs_labels_core(img_path=img_paths[index],
                                                      bbox=bboxes[index],
                                                      img_size=img_size,
                                                      normalizer=normalizer)
            # show(face)
            faces.append(face)
            labels.append(label)
        return faces, labels


def load_rough_imgs_occlus_core(img_path, bbox, img_size, normalizer=None):
    img = cv2.imread(img_path)
    face = cv2.resize(get_face(img, bbox, need_to_convert_to_int=True),
                      (img_size, img_size))
    if normalizer:
        face = normalizer.transform(face)
    label_path = os.path.splitext(img_path)[0] + ".opts"

    label = np.genfromtxt(label_path)
    return face, label


def load_rough_imgs_occlus(img_root, mat_file_name, img_size,
                           normalizer=None, chosen="random"):
    """Load rough imgs and occlus without normalization.

    :param img_root: img root dir
    :param mat_file_name:
    :param img_size:
    :param normalizer:
    :param chosen: whether to choose specific indices of dataset or just random

    :return chosen objs
    """
    img_paths, bboxes = load_basic_info(mat_file_name, img_root)
    if chosen == "random":
        length = len(img_paths)
        index = np.random.randint(0, length)
        return load_rough_imgs_occlus_core(img_path=img_paths[index],
                                           bbox=bboxes[index],
                                           img_size=img_size,
                                           normalizer=normalizer)
    else:
        faces = []
        labels = []
        for index in chosen:
            face, label = load_rough_imgs_occlus_core(img_path=img_paths[index],
                                                      bbox=bboxes[index],
                                                      img_size=img_size,
                                                      normalizer=normalizer)
            # show(face)
            faces.append(face)
            labels.append(label)
        return faces, labels


def draw_landmark(img, landmarks):
    for i in range(len(landmarks)):
        for idx, point in enumerate(landmarks):
            pos = (int(point[0]), int(point[1]))
            cv2.circle(img, pos, 5, color=(0, 255, 0))
    cv2.namedWindow("img", 2)
    cv2.imshow("img", img)
    cv2.waitKey(0)


def binary(num, threshold):
    num = 1 if num > threshold else 0
    return num


def extend(data_base, add_data):
    data = [_ for _ in data_base]
    data.extend(add_data)
    return data


def get_filenames(data_dir, listext, label_ext):
    img_list = []
    label_list = []
    for ext in listext:
        p = os.path.join(data_dir, ext)
        img_list.extend(glob.glob(p))

    for img in img_list:
        label_list.append(os.path.splitext(img)[0] + label_ext)

    return img_list, label_list
