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

def gen_n2c_stim_h5_pairs(save_dir, files, noise_name, mode, win=50, stride=20):  # 生成模拟噪声干净图像快对
    clean_h5_path = save_dir + '/n2c_' + noise_name + '_clean_' + mode + '.h5'
    noised_h5_path = save_dir + '/n2c_' + noise_name + '_noised_' + mode + '.h5'
    print(clean_h5_path, noised_h5_path)
    clean_h5 = h5py.File(clean_h5_path, 'w')
    noised_h5 = h5py.File(noised_h5_path, 'w')
    noise_type, sttv = noise_name.split('_')[0], int(noise_name.split('_')[1])
    num = 0
    for i in tqdm(range(len(files))):
        clean = cv2.imread(files[i])
        if noise_type == 'gaussian':
            noised = add_gaussian_noise(clean, sttv)
        elif noise_type == 'impluse':
            noised = add_impulse_noise(clean, sttv)
        elif noise_type == 'text':
            noised = add_text_noise(clean, sttv)
        elif noise_type == 'multi':
            noised = add_multi_noise(clean, sttv)
        if mode == 'train':
            clean_patches = image2patches(clean, win, stride)
            noised_patches = image2patches(noised, win, stride)
            for n in range(clean_patches.shape[0]):
                clean_data = clean_patches[n, :, :, :]
                noised_data = noised_patches[n, :, :, :]
                clean_h5.create_dataset(str(num), data=clean_data)
                noised_h5.create_dataset(str(num), data=noised_data)
                num += 1
        else:
            clean_h5.create_dataset(str(num), data=clean)
            noised_h5.create_dataset(str(num), data=noised)
            num += 1
    clean_h5.close()
    noised_h5.close()


def add_noise_to_video(vid_path, noise_name='multi_10'):  # 对视频增加模拟噪声
    cap = cv2.VideoCapture(vid_path)
    total_frame_num_VIDEO = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    dstSize = (int(frameWidth), int(frameHeight))

    noise_type, sttv = noise_name.split('_')[0], int(noise_name.split('_')[1])
    out = cv2.VideoWriter(vid_path.replace('.avi', '_' + noise_name + '.avi'),
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, dstSize)

    for idx in tqdm(range(total_frame_num_VIDEO - 1)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        _, clean = cap.read()

        if noise_type == 'gaussian':
            noised = add_gaussian_noise(clean, sttv)
        elif noise_type == 'impluse':
            noised = add_impulse_noise(clean, sttv)
        elif noise_type == 'text':
            noised = add_text_noise(clean, sttv)
        elif noise_type == 'multi':
            noised = add_multi_noise(clean, sttv)

        out.write(noised)
    out.release()


def add_noise_to_images(img_dir, noise_name='multi_10'):  # 对图像增加模拟噪声
    save_dir = img_dir + '_' + noise_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_paths = [i for i in Path(img_dir).glob('*.jpg')]
    noise_type, sttv = noise_name.split('_')[0], int(noise_name.split('_')[1])
    for img_path in tqdm(img_paths):
        clean = cv2.imread(str(img_path))
        if noise_type == 'gaussian':
            noised = add_gaussian_noise(clean, sttv)
        elif noise_type == 'impluse':
            noised = add_impulse_noise(clean, sttv)
        elif noise_type == 'text':
            noised = add_text_noise(clean, sttv)
        elif noise_type == 'multi':
            noised = add_multi_noise(clean, sttv)
        cv2.imwrite(save_dir + '/' + img_path.name, noised)


def gen_voc_h5(voc_dir, set_name, noise_name, gen_mode):  # 对VOC数据集增加模拟噪声
    if set_name is not None:
        with open(voc_dir + '/ImageSets/Main/' + set_name + '.txt', 'r') as f:
            lines = f.readlines()
        image_paths = [voc_dir + '/JPEGImages/' + i.strip() + '.jpg' for i in lines if i.strip() != '']
        save_dir = voc_dir + '/' + set_name + '_h5'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        image_paths = [str(i) for i in Path(voc_dir + '/JPEGImages').glob('*.jpg')]
        save_dir = voc_dir
    if gen_mode == 'test':
        gen_n2c_stim_h5_pairs(save_dir, image_paths, noise_name, 'test')
    else:
        random.shuffle(image_paths)
        test_paths = image_paths[:24]
        train_paths = [i for i in image_paths if i not in test_paths]
        gen_n2c_stim_h5_pairs(save_dir, train_paths, noise_name, 'train')
        gen_n2c_stim_h5_pairs(save_dir, test_paths, noise_name, 'test')

def gen_public_stim_h5(data_dir, dn_mode, data_mode, noise_name, win, stride=20, num=0):  # 准备公开的模拟去噪数据集，BSD等，噪声干净图像块对
    data_name = Path(data_dir).name
    save_dir = data_dir
    noise_type, sttv = noise_name.split('_')[0], int(noise_name.split('_')[1])
    suffixs = ['jpg', 'JPG', 'bmp', 'JPEG', 'png', 'jpeg']
    clean_paths = [i for i in Path(data_dir + '/images').glob('*.*') if i.name.split('.')[-1] in suffixs]
    h5_path_clean = save_dir + '/' + dn_mode + '_' + noise_name + '_' + data_name + '_clean_' + data_mode + '.h5'
    print('h5_path_clean:', h5_path_clean)
    h5_clean = h5py.File(h5_path_clean, 'w')
    h5_noised = h5py.File(h5_path_clean.replace('clean', 'noised'), 'w')
    data_num = 0
    for clean_path in tqdm(clean_paths):
        clean_ = cv2.imread(str(clean_path))
        if dn_mode == 'n2c':
            clean = clean_
            if noise_type == 'gaussian':
                noised = add_gaussian_noise(clean_, sttv)
            elif noise_type == 'impluse':
                noised = add_impulse_noise(clean_, sttv)
            elif noise_type == 'text':
                noised = add_text_noise(clean_, sttv)
        elif dn_mode == 'n2n':
            if noise_type == 'gaussian':
                clean = add_gaussian_noise(clean_, sttv)
                noised = add_gaussian_noise(clean_, sttv)
            elif noise_type == 'impluse':
                clean = add_gaussian_noise(clean_, sttv)
                noised = add_impulse_noise(clean_, sttv)
            elif noise_type == 'text':
                clean = add_gaussian_noise(clean_, sttv)
                noised = add_text_noise(clean_, sttv)
        if data_mode == 'train':
            if num == 0:
                clean_patches = image2patches(clean, win, stride)
                noised_patches = image2patches(noised, win, stride)
            else:
                clean_patches, noised_patches = image2patches_random(clean, noised, win, num)
            assert clean_patches.shape == noised_patches.shape
            patch_num_per_img = clean_patches.shape[0]
            # print(patch_num_per_img)
            for n in range(patch_num_per_img):
                clean_data = clean_patches[n, :, :, :]
                noised_data = noised_patches[n, :, :, :]
                h5_clean.create_dataset(str(data_num), data=clean_data)
                h5_noised.create_dataset(str(data_num), data=noised_data)
                data_num += 1
        elif data_mode == 'test':
            h5_clean.create_dataset(str(data_num), data=clean)
            h5_noised.create_dataset(str(data_num), data=noised)
            data_num += 1
    h5_clean.close()
    h5_noised.close()
    print('data_num:{}'.format(data_num))
