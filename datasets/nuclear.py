import os
import cv2
import torch
from matplotlib import pyplot as plt
import numpy as np
import h5py
import operator
from pathlib import Path
import random
import math
from tqdm import tqdm
from torch.utils.data import Dataset
from data import neighbor2neighbor


def gen_nuclear_h5(data_dir, save_dir, win=50, stride=20):  # 准备真实核辐射去噪数据集，噪声干净图像块对
    clean_test = h5py.File(save_dir + '/n2c_nuclear_clean_test.h5', 'w')
    noised_test = h5py.File(save_dir + '/n2c_nuclear_noised_test.h5', 'w')

    clean_train = h5py.File(save_dir + '/n2c_nuclear_clean_train.h5', 'w')
    noised_train = h5py.File(save_dir + '/n2c_nuclear_noised_train.h5', 'w')

    clean_paths = [i for i in Path(data_dir).rglob('*.png') if '0.png' == i.name]
    train_num = 0
    test_num = 0
    for clean_path in tqdm(clean_paths):
        clean = cv2.imread(str(clean_path))
        noised1 = cv2.imread(str(clean_path).replace('0.png', '1.png'))
        noised2 = cv2.imread(str(clean_path).replace('0.png', '2.png'))
        noised3 = cv2.imread(str(clean_path).replace('0.png', '3.png'))
        noised4 = cv2.imread(str(clean_path).replace('0.png', '4.png'))

        clean_test.create_dataset(str(test_num), data=clean)
        noised_test.create_dataset(str(test_num), data=noised4)
        test_num += 1

        clean_patches = image2patches(clean, win, stride)
        patch_num_per_img = clean_patches.shape[0]

        noiseds = [noised1, noised2, noised3, noised4]
        for noised in noiseds:
            noised_patches = image2patches(noised, win, stride)
            for n in range(patch_num_per_img):
                clean_data = clean_patches[n, :, :, :]
                noised_data = noised_patches[n, :, :, :]
                clean_train.create_dataset(str(train_num), data=clean_data)
                noised_train.create_dataset(str(train_num), data=noised_data)
                train_num += 1


def gen_nuclear_h5_n2n(data_dir, save_dir, k=2, pairs_per_img=4):
    clean_test = h5py.File(save_dir + '/n2n_nuclear_clean_test.h5', 'w')
    noised_test = h5py.File(save_dir + '/n2n_nuclear_noised_test.h5', 'w')
    clean_train = h5py.File(save_dir + '/n2n_nuclear_clean_train.h5', 'w')
    noised_train = h5py.File(save_dir + '/n2n_nuclear_noised_train.h5', 'w')
    clean_paths = [i for i in Path(data_dir).rglob('*.png') if '0.png' == i.name]
    train_num = 0
    test_num = 0
    for clean_path in tqdm(clean_paths):
        clean = cv2.imread(str(clean_path))
        noised1 = cv2.imread(str(clean_path).replace('0.png', '1.png'))
        noised2 = cv2.imread(str(clean_path).replace('0.png', '2.png'))
        noised3 = cv2.imread(str(clean_path).replace('0.png', '3.png'))

        noised4 = cv2.imread(str(clean_path).replace('0.png', '4.png'))
        clean_test.create_dataset(str(test_num), data=clean)
        noised_test.create_dataset(str(test_num), data=noised4)
        test_num += 1

        # clean_patches = image2patches(clean, win, stride)

        patch_num_per_img = k
        noiseds = [noised1, noised2, noised3]
        for noised in noiseds:
            down_sample1s
            down_sample1, down_sample2 = neighbor2neighbor(noised, k)

            for n in range(patch_num_per_img):
                clean_data = clean_patches[n, :, :, :]
                noised_data = noised_patches[n, :, :, :]
                clean_train.create_dataset(str(train_num), data=clean_data)
                noised_train.create_dataset(str(train_num), data=noised_data)
                train_num += 1
