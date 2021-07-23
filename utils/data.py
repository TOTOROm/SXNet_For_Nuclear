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

MAX = math.pow(2, 8)
# MAX = math.pow(2, 14)  # 归一化的分母，14位用于RAW图


def tensorNCHW_BGR_to_npHWC_RGB(tensor):
    tensor = tensor.cpu().detach()
    assert tensor.shape[0] == 1
    tensor = tensor.squeeze(0)
    n2 = tensor.permute(1, 2, 0)  # CHW to HWC
    if torch.max(n2) <= 1:
        n2 *= 255.0
        n2 = torch.clamp(n2, 0, 255)
    n2 = np.array(n2).astype('uint8')
    n2 = cv2.cvtColor(n2, cv2.COLOR_BGR2RGB)
    return n2


def add_noise_tensorNCHW(tensor, specificN=None):
    assert torch.max(tensor) <= 1
    if specificN is None:
        noise = torch.zeros(tensor.size())
        stdN = np.random.uniform(0, 50, size=noise.size()[0])
        for n in range(noise.size()[0]):
            sizeN = noise[0, :, :, :].size()
            noise[n, :, :, :] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n] / 255.)
    else:
        noise = torch.FloatTensor(tensor.size()).normal_(mean=0, std=specificN / 255.)
    noisy = tensor + noise
    noisy = torch.clamp(noisy, 0., 1.)
    return noisy


def vis_IN_OUT(noisy, noisy_output):
    noisy_np = tensorNCHW_BGR_to_npHWC_RGB(noisy)
    fulloutput_np = tensorNCHW_BGR_to_npHWC_RGB(noisy_output)
    plt.subplot(121)
    plt.imshow(noisy_np)
    plt.axis('off')
    plt.title('Input ' + str(noisy_np.shape))
    plt.subplot(122)
    plt.imshow(fulloutput_np)
    plt.axis('off')
    plt.title('Output ' + str(noisy_np.shape))
    plt.show()


def vis_IN_OUT_GT(noisy, noisy_output, GT):
    noisy_np = tensorNCHW_BGR_to_npHWC_RGB(noisy)
    fulloutput_np = tensorNCHW_BGR_to_npHWC_RGB(noisy_output)
    gt_np = tensorNCHW_BGR_to_npHWC_RGB(GT)
    plt.subplot(131)
    plt.imshow(noisy_np)
    plt.axis('off')
    plt.title('Input ' + str(noisy_np.shape))
    plt.subplot(132)
    plt.imshow(fulloutput_np)
    plt.axis('off')
    plt.title('Output ' + str(fulloutput_np.shape))
    plt.subplot(133)
    plt.imshow(gt_np)
    plt.axis('off')
    plt.title('GT ' + str(gt_np.shape))
    plt.show()


def vis_IN_OUT_source_target(noisy, noisy_output, noisy_sub1, noisy_sub2):
    noisy_np = tensorNCHW_BGR_to_npHWC_RGB(noisy)
    fulloutput_np = tensorNCHW_BGR_to_npHWC_RGB(noisy_output)
    red_np = tensorNCHW_BGR_to_npHWC_RGB(noisy_sub1)
    blue_np = tensorNCHW_BGR_to_npHWC_RGB(noisy_sub2)
    plt.subplot(221)
    plt.imshow(noisy_np)
    plt.axis('off')
    plt.title('Input ' + str(noisy_np.shape))
    plt.subplot(222)
    plt.imshow(fulloutput_np)
    plt.axis('off')
    plt.title('Output ' + str(noisy_np.shape))
    plt.subplot(223)
    plt.imshow(red_np)
    plt.axis('off')
    plt.title('PatchIn ' + str(red_np.shape))
    plt.subplot(224)
    plt.imshow(blue_np)
    plt.axis('off')
    plt.title('PatchOut ' + str(blue_np.shape))
    plt.show()


class dataset_img2img(Dataset):  # 用于pytorch的dataloader
    def __init__(self, source_h5_path, target_h5_path, break_id=0):  # break_id是拿来方便测试的，默认等于0，也就是没用
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

        source = torch.from_numpy(np.float32(source / MAX)).permute(2, 0, 1)  # 不管是多少位的图，都用float32的tensor来训练
        target = torch.from_numpy(np.float32(target / MAX)).permute(2, 0, 1)

        return source, target


class dataset_single_h5(Dataset):
    def __init__(self, source_h5_path):
        super(dataset_single_h5, self).__init__()
        self.source_h5 = h5py.File(source_h5_path, 'r')
        self.keys = list(self.source_h5.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        source = np.array(self.source_h5[key])  # HWC
        source = torch.from_numpy(source).permute(2, 1, 0).float()  # HWC to CHW float32 , torch tensor
        return source


def image2patches(img, win, stride):  # 把图像转换位图像块（滑窗法）
    if len(img.shape) == 2:
        img = np.array([img]).transpose((1, 2, 0))
        # print(img.shape)
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


# def neighbor2neighbor(img, k=2):
#     h, w, c = img.shape
#     h2, w2 = h // k, w // k
#     down_sample1, down_sample2 = [], []
#     for i in range(0, h, 2):
#         for j in range(0, h, 2):
#             tmp = img[i:i + k, j:j + k, :]
#             random_pixel_ids = np.random.choice(k * k, 2, replace=False)  # 随机选两个不重复的坐标
#             down_sample1.append(tmp[random_pixel_ids[0] // k][random_pixel_ids[0] % k])
#             down_sample2.append(tmp[random_pixel_ids[1] // k][random_pixel_ids[1] % k])
#     down_sample1 = np.array(down_sample1).astype(img.dtype)
#     down_sample2 = np.array(down_sample2).astype(img.dtype)
#     down_sample1.reshape((h2, w2, c))
#     down_sample2.reshape((h2, w2, c))
#     return down_sample1, down_sample2


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


def image2patches_random2(img1, win, num):  # 把图像转换位图像块（随机裁剪法）
    h, w, _ = img1.shape
    assert win < h and win < w
    rand_tl_x = random.randint(0, w - win)
    rand_tl_y = random.randint(0, h - win)
    br_y = rand_tl_y + win
    br_x = rand_tl_x + win
    patches1 = []
    n = 0
    while True:
        if br_x < w and br_y < h:
            rand_patch1 = img1[rand_tl_y: br_y, rand_tl_x:br_x]
            patches1.append(rand_patch1)
            n += 1
            if n >= num:
                break
    return np.array(patches1)


def read_raw_by_np(raw_path, shape=(256, 336), dtype='uint16'):  # 读raw图为ndarray类型
    raw_data = np.fromfile(raw_path, dtype=dtype)  # 按照uint16来读和处理
    raw_data_real = raw_data
    imgData = raw_data_real.reshape(shape[0], shape[1])  # 利用numpy中array的reshape函数将读取到的数据进行重新排列。

    imgData = np.expand_dims(imgData, axis=2)
    # print('sss', imgData.shape)
    return imgData


def gen_clean_for_a_scene(scene_raw_dir):  # 对某个文件夹的raw图求均，获得其干净图
    raw_paths = [i for i in Path(scene_raw_dir).glob('*.raw')]

    raw_first = read_raw_by_np(str(raw_paths[0]))
    raw_shape = raw_first.shape
    raw_type = raw_first.dtype

    raw_clean = np.zeros(raw_shape, dtype='float64')
    for i in tqdm(range(len(raw_paths))):
        raw_tmp = read_raw_by_np(str(raw_paths[i]))
        raw_clean = raw_clean + raw_tmp
    raw_clean = raw_clean / len(raw_paths)
    raw_clean = raw_clean.astype(raw_type)

    raw_clean.tofile(str(Path(scene_raw_dir).parent) + '/' + Path(scene_raw_dir).name + '_clean.raw')

    return raw_clean


def psnr(img1, img2):  # 求PSNR
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def vis_h5(noised_h5_path, data_type='uint8'):  # 可视化数据集，以验证数据集
    assert 'noised' in noised_h5_path
    clean_h5_path = noised_h5_path.replace('noised', 'clean')
    if 'test' in noised_h5_path:
        break_id = 0
    else:
        break_id = 18
    dataset = dataset_img2img(noised_h5_path, clean_h5_path, break_id)
    print('len(dataset):', len(dataset))
    P = []

    for n, pair in enumerate(dataset):
        noised = np.array(pair[0].permute(1, 2, 0) * MAX).astype(data_type)
        clean = np.array(pair[1].permute(1, 2, 0) * MAX).astype(data_type)

        if noised.shape[2] == 1:
            noised = noised.squeeze()
            clean = clean.squeeze()

        p = psnr(clean, noised)
        P.append(p)
        print('psnr', p)

        # print(noised.shape, noised.dtype, type(noised), np.mean(noised))
        plt.subplot(121)
        plt.title('Noised')
        plt.imshow(noised)
        plt.axis('off')
        plt.subplot(122)
        plt.title('Clean')
        plt.imshow(clean)
        plt.axis('off')
        plt.show(0.1)
        # if n % 2 == 0:
        #     cv2.imwrite(str(n) + '_clean.png', clean, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # else:
        #     cv2.imwrite(str(n) + '_noised.png', noised, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    print('mPSNR:{}'.format(np.mean(P)))


def data_augmentation(data, mode):
    # print(data.shape)
    # out = np.transpose(image, (1, 2, 0))
    # out = np.squeeze(data, axis=0)
    if mode == 1:
        # flip up and down
        out = np.flipud(data)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(data)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(data)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(data, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(data, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(data, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(data, k=3)
        out = np.flipud(out)
    else:  # mode=0
        out = data
    return out


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

    # gen_infread_h5_denoise_raw('/home/ipsg/code/sx/datasets/infread/raws/cover_low')
    # vis_h5('/home/ipsg/code/sx/datasets/infread/raws/cover_low/n2c_infread_noised_test.h5')

    vis_h5('/home/SENSETIME/sunxin/2_myrepos/data/lowlight_denoise/h5/n2c_lowlight_noised_test.h5')

