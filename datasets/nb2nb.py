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

import sys, os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

from utils.data import image2patches_random2, data_augmentation, dataset_single_h5


def gen_nb2nb_h5(data_dir, win=256, rand_crop_n=1000, aug_times=1):
    save_dir = data_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    paths = [i for i in Path(data_dir).glob('*.jpg')]
    paths_train = paths
    train = h5py.File(save_dir + '/n2c_nb2nb_train.h5', 'w')
    train_num = 0
    for path in tqdm(paths_train):
        noisy = cv2.imread(str(path))
        patches = image2patches_random2(noisy, win, rand_crop_n)
        patch_num_per_img = patches.shape[0]
        for n in range(patch_num_per_img):
            data = patches[n, :, :, :].copy()
            # print('ss',data.shape)
            # plt.imshow(data)
            # plt.pause(0.1)
            train.create_dataset(str(train_num), data=data)
            train_num += 1
            for m in range(aug_times):
                data_aug = data_augmentation(data.copy(), np.random.randint(1, 8))
                # print('ss', data_aug.shape)
                # plt.imshow(data_aug)
                # plt.pause(0.1)
                train.create_dataset(str(train_num), data=data_aug)
                train_num += 1
    train.close()

    test = h5py.File(save_dir + '/n2c_nb2nb_test.h5', 'w')
    paths_test = paths
    test_num = 0
    for path in tqdm(paths_test):
        noisy = cv2.imread(str(path))
        patches2 = image2patches_random2(noisy, 1024, 4)
        patch_num_per_img2 = patches2.shape[0]
        for n in range(patch_num_per_img2):
            data2 = patches2[n, :, :, :]
            test.create_dataset(str(test_num), data=data2)
            test_num += 1
    test.close()
    print('train_num:{}, test_num:{}'.format(train_num, test_num))


if __name__ == '__main__':
    # gen_nb2nb_h5('/home/SENSETIME/sunxin/data/front')
    gen_nb2nb_h5('/home/ipsg/code/sx/datasets/front')

    dataset = dataset_single_h5('/home/ipsg/code/sx/datasets/front/n2c_nb2nb_train.h5')
    print('len(dataset):', len(dataset))
    # for n, data in enumerate(dataset):
    #     print('ss', data.shape)
    #     plt.imshow(data)
    #     plt.pause(0.1)
