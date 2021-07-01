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

def gen_data_from_videos(data_root='E:/datasets/infread/videos', hw_range=(69, 480, 0, 640)):  # 将红外视频直接保存为无损图像
    H_dir = data_root + '/H'  # 高增益的视频
    L_dir = data_root + '/L'  # 低增益的视频
    dirs = [H_dir, L_dir]
    frame_nums = []
    for dir in dirs:
        assert 'videos' in str(dir)
        for vid_path in Path(dir).glob('*.avi'):
            cap = cv2.VideoCapture(str(vid_path))
            total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            frame_nums.append(total_frame_num)
    frame_num = min(frame_nums)
    if frame_num > 999:
        frame_num = 999

    for dir in dirs:
        for vid_path in Path(dir).glob('*.avi'):
            save_dir1 = str(dir).replace('videos', 'images') + '/' + vid_path.name[:-4]
            if not os.path.exists(save_dir1):
                os.makedirs(save_dir1)
            cap = cv2.VideoCapture(str(vid_path))
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > frame_num
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _, first = cap.read()
            first = first[hw_range[0]:hw_range[1], hw_range[2]:hw_range[3]]
            mean = np.zeros(first.shape, dtype='float64')
            for id in tqdm(range(frame_num)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, id)
                _, frame = cap.read()
                frame = frame[hw_range[0]:hw_range[1], hw_range[2]:hw_range[3]]
                cv2.imwrite(save_dir1 + '/noised_' + vid_path.name[:-4] + '_' + '{:03d}'.format(id) + '_.png',
                            frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                mean = mean + frame
            for c in range(3):
                mean[:, :, c] /= (frame_num)
            mean = mean.astype('uint8')
            cv2.imwrite(save_dir1 + '/clean_' + vid_path.name[:-4] + '.png', mean,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def gen_infread_h5_enhance(data_root):  # 将红外无损图像保存为噪声干净图像块对数据集。对红外增强而言，干净表示高增益，噪声表示低增益
    save_dir = data_root
    clean_test = h5py.File(save_dir + '/n2c_infreadEN256_clean_test.h5', 'w')
    noised_test = h5py.File(save_dir + '/n2c_infreadEN256_noised_test.h5', 'w')
    clean_train = h5py.File(save_dir + '/n2c_infreadEN256_clean_train.h5', 'w')
    noised_train = h5py.File(save_dir + '/n2c_infreadEN256_noised_train.h5', 'w')

    H_root = data_root + '/H'
    train_num = 0
    test_num = 0
    for img_path in tqdm(Path(H_root).rglob('*.png')):
        if 'clean' in img_path.name:
            continue
        else:
            clean = cv2.imread(str(img_path))
            noised = cv2.imread(str(img_path).replace('H', 'L'))

        if '000' in img_path.name or '128' in img_path.name:
            clean_test.create_dataset(str(test_num), data=clean)
            noised_test.create_dataset(str(test_num), data=noised)
            test_num += 1
        else:
            win, stride = 256, 40
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
    clean_test.close()
    noised_test.close()

    print('train_num:{}'.format(train_num))
    print('test_num:{}'.format(test_num))


def gen_infread_h5_denoise(data_root):  # 将红外无损图像保存为噪声干净图像块对数据集。对红外去噪而言，干净表示均值基准图，噪声表示原始图
    save_dir = data_root
    clean_test = h5py.File(save_dir + '/n2c_infreadDNL_clean_test.h5', 'w')
    noised_test = h5py.File(save_dir + '/n2c_infreadDNL_noised_test.h5', 'w')
    clean_train = h5py.File(save_dir + '/n2c_infreadDNL_clean_train.h5', 'w')
    noised_train = h5py.File(save_dir + '/n2c_infreadDNL_noised_train.h5', 'w')

    train_num = 0
    test_num = 0
    for img_path in tqdm(Path(data_root).rglob('*.png')):
        if 'clean' in img_path.name:
            continue
        else:
            clean_name = img_path.name.replace('noised', 'clean')[:11] + '.png'
            clean = cv2.imread(str(img_path.parent) + '/' + clean_name)
            noised = cv2.imread(str(img_path))

        if '000' in img_path.name or '128' in img_path.name:
            clean_test.create_dataset(str(test_num), data=clean)
            noised_test.create_dataset(str(test_num), data=noised)
            test_num += 1
        else:
            win, stride = 50, 40
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
    clean_test.close()
    noised_test.close()

    print('train_num:{}'.format(train_num))
    print('test_num:{}'.format(test_num))

def gen_infread_h5_denoise_raw(data_root, test_num_each_scene=2, win=50, stride=10):  # denoise for RAW
    assert 'low' in data_root or 'high' in data_root
    save_dir = data_root

    clean_test_L = h5py.File(save_dir + '/n2c_infread_clean_test.h5', 'w')
    noised_test_L = h5py.File(save_dir + '/n2c_infread_noised_test.h5', 'w')
    clean_train_L = h5py.File(save_dir + '/n2c_infread_clean_train.h5', 'w')
    noised_train_L = h5py.File(save_dir + '/n2c_infread_noised_train.h5', 'w')

    train_num_L = 0
    test_num_L = 0

    scene_dirs = [i for i in Path(data_root).iterdir() if i.is_dir()]  # s1~s8的场景

    for scene_dir in scene_dirs:

        low_clean = gen_clean_for_a_scene(str(scene_dir))


        low_paths = [i for i in Path(str(scene_dir)).glob('*.raw')]
        if len(low_paths) <= test_num_each_scene:
            continue

        random.shuffle(low_paths)

        low_paths_train = low_paths[test_num_each_scene:]
        low_paths_test = low_paths[:test_num_each_scene]

        for low_path_train in low_paths_train:
            clean = low_clean
            noised = read_raw_by_np(str(low_path_train))
            clean_patches = image2patches(clean, win, stride)
            patch_num_per_img = clean_patches.shape[0]
            noised_patches = image2patches(noised, win, stride)
            for n in range(patch_num_per_img):
                clean_data = clean_patches[n, :, :, :]
                noised_data = noised_patches[n, :, :, :]
                clean_train_L.create_dataset(str(train_num_L), data=clean_data)
                noised_train_L.create_dataset(str(train_num_L), data=noised_data)
                train_num_L += 1

        for low_path_test in low_paths_test:
            clean = low_clean
            noised = read_raw_by_np(str(low_path_test))
            clean_test_L.create_dataset(str(test_num_L), data=clean)
            noised_test_L.create_dataset(str(test_num_L), data=noised)
            test_num_L += 1


    clean_train_L.close()
    noised_train_L.close()
    clean_test_L.close()
    noised_test_L.close()

    print('train_num_L:{}'.format(train_num_L))
    print('test_num_L:{}'.format(test_num_L))
