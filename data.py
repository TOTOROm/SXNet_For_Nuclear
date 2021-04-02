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


class dataset_img2img(Dataset):  # 用于pytorch的dataloader
    def __init__(self, source_h5_path, target_h5_path, break_id=0):
        super(dataset_img2img, self).__init__()
        self.source_h5 = h5py.File(source_h5_path, 'r')
        self.target_h5 = h5py.File(target_h5_path, 'r')
        assert operator.eq(list(self.source_h5.keys()), list(self.target_h5.keys()))
        if break_id > 0:
            self.keys = list(self.source_h5.keys())[:break_id]
        else:
            self.keys = list(self.source_h5.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        source = np.array(self.source_h5[key])
        target = np.array(self.target_h5[key])
        source = torch.from_numpy(np.float32(source / 255)).permute(2, 0, 1)
        target = torch.from_numpy(np.float32(target / 255)).permute(2, 0, 1)
        return source, target


def image2patches(img, win, stride):  # 把图像转换位图像块（滑窗法）
    h, w, _ = img.shape
    assert win < h and win < w
    patches = []
    for i in range(0, h - win + 1, stride):
        for j in range(0, w - win + 1, stride):
            # print(img)
            patch = img[i:i + win, j:j + win, :]
            # print(patch)
            patches.append(patch)
    # print(patches)
    return np.array(patches)


def image2patches_random(img1, img2, win, num):  # 把图像转换位图像块（随机裁剪法）
    assert img1.shape == img2.shape
    h, w, _ = img1.shape
    assert win < h and win < w
    rand_tl_x = random.randint(0, w - win)
    rand_tl_y = random.randint(0, h - win)
    br_y = rand_tl_y + win
    br_x = rand_tl_x + win
    patch_pairs = []
    n = 0
    while True:
        if br_x < w and br_y < h:
            rand_patch1 = img1[rand_tl_y: br_y, rand_tl_x:br_x]
            rand_patch2 = img2[rand_tl_y: br_y, rand_tl_x:br_x]
            patch_pairs.append((rand_patch1, rand_patch2))
            n += 1
            if n >= num:
                break
    random.shuffle(patch_pairs)
    patches1, patches2 = [], []
    for pair in patch_pairs:
        patches1.append(pair[0])
        patches2.append(pair[1])
    return np.array(patches1), np.array(patches2)


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


def psnr(img1, img2):  # 求PSNR
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def std(img, hw_range):  # 求标准差
    if hw_range is not None:
        img = img[hw_range[0]:hw_range[1], hw_range[2]:hw_range[3], :]
    h, w, c = img.shape
    if c == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, var_ = cv2.meanStdDev(img)
    std_ = math.sqrt(var_)
    return std_


def vis_h5(noised_h5_path):  # 可视化数据集，以验证数据集
    assert 'noised' in noised_h5_path
    clean_h5_path = noised_h5_path.replace('noised', 'clean')
    if 'test' in noised_h5_path:
        break_id = 0
    else:
        break_id = 18
    dataset = dataset_img2img(noised_h5_path, clean_h5_path, break_id)
    print('len(dataset):', len(dataset))
    P = []
    S1 = []
    S2 = []
    for n, pair in enumerate(dataset):
        noised = np.array(pair[0].permute(1, 2, 0) * 255).astype('uint8')
        clean = np.array(pair[1].permute(1, 2, 0) * 255).astype('uint8')

        # noised = np.array(pair[0].permute(1, 2, 0) * 255)
        # clean = np.array(pair[1].permute(1, 2, 0) * 255)
        # from noise_estimation_2015 import noise_estimate

        # from noised_estimate_2 import noiseLevelEstimation
        # noised = cv2.cvtColor(noised, cv2.COLOR_BGR2GRAY)
        # clean = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
        # patchSize = 7
        # confidenceLevel = 1 - 1e-6  # choose close to 1
        # numIteration = 3
        # EV1 = noiseLevelEstimation(noised, patchSize, confidenceLevel, numIteration)
        # s1 = math.sqrt(EV1)
        # EV2 = noiseLevelEstimation(clean, patchSize, confidenceLevel, numIteration)
        # s2 = math.sqrt(EV2)
        # s1 = noise_estimate(noised, 8)
        # s2 = noise_estimate(clean, 8)

        # hw_range = (10, 20, 10, 20)
        # # hw_range = None
        # s1 = std(noised, hw_range)
        # s2 = std(clean, hw_range)
        # p = psnr(clean, noised)
        # # print('PSNR:{}, NoisedStd:{}, CleanStd:{}'.format(p, s1, s2))
        # P.append(p)
        # S1.append(s1)
        # S2.append(s2)

        plt.subplot(121)
        plt.title('Noised')
        plt.imshow(cv2.cvtColor(noised, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.subplot(122)
        plt.title('Clean')
        plt.imshow(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        # if n % 2 == 0:
        #     cv2.imwrite(str(n) + '_clean.png', clean, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # else:
        #     cv2.imwrite(str(n) + '_noised.png', noised, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # print('mPSNR:{}, mNoisedStd:{}, mCleanStd:{}'.format(np.mean(P), np.mean(S1), np.mean(S2)))


if __name__ == '__main__':
    # add_noise_to_images()
    # add_noise_to_video()

    # gen_infread_h5_enhance(data_root='/home/ipsg/code/sx/datasets/infread/images')
    # gen_infread_h5_denoise(data_root='/home/ipsg/code/sx/datasets/infread/images/L')

    # gen_voc_h5('/home/ipsg/code/sx/datasets/nuclear_cold_tiny', set_name=None, noise_name='gaussian_75',
    #            gen_mode='both')

    # gen_polyu_h5('/home/ipsg/code/sx/datasets/polyu200c/images')

    # gen_public_stim_h5(data_dir='/home/ipsg/code/sx/datasets/public_dn/pristine',
    #                    dn_mode='n2c',
    #                    data_mode='train',
    #                    noise_name='gaussian_50',
    #                    win=50,
    #                    stride=25, num=0)

    # vis_h5('/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN256_noised_test.h5')
    vis_h5('/home/ipsg/code/sx/datasets/infread/images/L/n2c_infreadDNL_noised_test.h5')
