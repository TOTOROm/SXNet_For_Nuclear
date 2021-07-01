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

from utils.add_noise import add_impulse_noise, add_text_noise, add_gaussian_noise, add_multi_noise

def gen_polyu_h5(data_dir, win=50, stride=20):  # 准备POLYU真实去噪数据集，噪声干净图像块对
    save_dir = data_dir.replace('images', '')
    clean_paths = [i for i in Path(data_dir).glob('*.JPG') if 'mean' in i.name]
    random.shuffle(clean_paths)
    clean_paths_test = clean_paths[:24]
    clean_paths_train = [i for i in clean_paths if i not in clean_paths_test]
    clean_train = h5py.File(save_dir + '/n2c_polyu_clean_train.h5', 'w')
    noised_train = h5py.File(save_dir + '/n2c_polyu_noised_train.h5', 'w')
    train_num = 0
    for clean_path in tqdm(clean_paths_train):
        clean = cv2.imread(str(clean_path))
        noised = cv2.imread(str(clean_path).replace('mean', 'real'))
        clean_patches = image2patches(clean, win, stride)
        patch_num_per_img = clean_patches.shape[0]
        noised_patches = image2patches(noised, win, stride)
        for n in range(patch_num_per_img):
            clean_data = clean_patches[n, :, :, :]
            noised_data = noised_patches[n, :, :, :]
            clean_train.create_dataset(str(train_num), data=clean_data)
            noised_train.create_dataset(str(train_num), data=noised_data)
            train_num += 1
    clean_train.close()
    noised_train.close()
    clean_test = h5py.File(save_dir + '/n2c_polyu_clean_test.h5', 'w')
    noised_test = h5py.File(save_dir + '/n2c_polyu_noised_test.h5', 'w')
    test_num = 0
    for clean_path in tqdm(clean_paths_test):
        clean = cv2.imread(str(clean_path))
        noised = cv2.imread(str(clean_path).replace('mean', 'real'))
        clean_test.create_dataset(str(test_num), data=clean)
        noised_test.create_dataset(str(test_num), data=noised)
        test_num += 1
    clean_test.close()
    noised_test.close()
    print('train_num:{}, test_num:{}'.format(train_num, test_num))
