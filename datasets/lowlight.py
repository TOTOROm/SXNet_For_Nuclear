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
from utils.data import image2patches, vis_h5


def gen_lowlight_h5(data_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    clean_test = h5py.File(save_dir + '/n2c_lowlight_clean_test.h5', 'w')
    noised_test = h5py.File(save_dir + '/n2c_lowlight_noised_test.h5', 'w')

    clean_train = h5py.File(save_dir + '/n2c_lowlight_clean_train.h5', 'w')
    noised_train = h5py.File(save_dir + '/n2c_lowlight_noised_train.h5', 'w')

    clean_paths = [i for i in Path(data_dir).glob('*.jpg') if 'denoise.jpg' in str(i)]
    train_num = 0
    test_num = 0
    for clean_path in tqdm(clean_paths):
        clean = cv2.imread(str(clean_path))
        noised = cv2.imread(str(clean_path).replace('denoise.jpg', 'mfnr.jpg'))
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
        noised = cv2.cvtColor(noised, cv2.COLOR_BGR2GRAY)
        # plt.subplot(121)
        # plt.title('Noised')
        # plt.imshow(noised)
        # plt.axis('off')
        # plt.subplot(122)
        # plt.title('Clean')
        # plt.imshow(clean)
        # plt.axis('off')
        # plt.show()
        h, w = clean.shape
        assert clean.shape == noised.shape

        clean1 = clean[:h // 2, :w // 2]
        clean2 = clean[h // 2:, w // 2:]
        clean3 = clean[h // 2:, :w // 2]
        clean4 = clean[:h // 2, w // 2:]
        assert clean1.shape == clean2.shape
        assert clean3.shape == clean2.shape
        assert clean4.shape == clean3.shape

        noised1 = noised[:h // 2, :w // 2]
        noised2 = noised[h // 2:, w // 2:]
        noised3 = noised[h // 2:, :w // 2]
        noised4 = noised[:h // 2, w // 2:]
        assert noised1.shape == noised2.shape
        assert noised3.shape == noised2.shape
        assert noised4.shape == noised3.shape

        cleans = [clean1, clean2, clean3, clean4]
        noiseds = [noised1, noised2, noised3, noised4]

        for i in range(len(cleans)):
            clean_patches = image2patches(cleans[i], win=48, stride=16)
            noised_patches = image2patches(noiseds[i], win=48, stride=16)
            # print(cleans[i].shape, clean_patches.shape)
            patch_num_per_img = clean_patches.shape[0]
            for n in range(patch_num_per_img):
                clean_data = clean_patches[n, :, :, :]
                noised_data = noised_patches[n, :, :, :]
                clean_train.create_dataset(str(train_num), data=clean_data)
                noised_train.create_dataset(str(train_num), data=noised_data)
                train_num += 1

        for i in range(len(cleans)):
            clean_patches = image2patches(cleans[i], win=512, stride=500)
            noised_patches = image2patches(noiseds[i], win=512, stride=500)
            patch_num_per_img = clean_patches.shape[0]
            for n in range(patch_num_per_img):
                clean_data = clean_patches[n, :, :, :]
                noised_data = noised_patches[n, :, :, :]
                clean_test.create_dataset(str(test_num), data=clean_data)
                noised_test.create_dataset(str(test_num), data=noised_data)
                test_num += 1

    print('train_num', train_num)
    print('test_num', test_num)


if __name__ == '__main__':
    # gt_bike = cv2.imread('/home/SENSETIME/sunxin/2_myrepos/data/lowlight_denoise/1623960484005_denoise.jpg')
    # in_bike = cv2.imread('/home/SENSETIME/sunxin/2_myrepos/data/lowlight_denoise/1623960484005_mfnr.jpg')
    # gt_face = cv2.imread('/home/SENSETIME/sunxin/2_myrepos/data/lowlight_denoise/1623961160647_denoise.jpg')
    # in_face = cv2.imread('/home/SENSETIME/sunxin/2_myrepos/data/lowlight_denoise/1623961160647_mfnr.jpg')

    data_dir = '/home/SENSETIME/sunxin/2_myrepos/data/blackshark/msdct/now'
    save_dir = '/home/SENSETIME/sunxin/2_myrepos/data/blackshark/h5'
    gen_lowlight_h5(data_dir, save_dir)
