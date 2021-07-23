import os
from argparse import ArgumentParser
import cv2
import json
import time
from utils.metric import *
# from data import dataset_img2img
from pathlib import Path
# from train_h5 import choose_model
from ptflops import get_model_complexity_info
from matplotlib import pyplot as plt
import math
import os
import random
import shutil
import json
from argparse import ArgumentParser
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utils.data import dataset_img2img
from utils.metric import batch_PSNR, ssim
from utils.choices import choose_loss, choose_model

MAX = math.pow(2, 8)


# MAX = math.pow(2, 14)  # 归一化的分母，14位用于RAW图


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str,
                        default='/home/SENSETIME/sunxin/2_myrepos/data/blackshark/DnCNN/DnCNN-bz64_ep120_l1')
    parser.add_argument("--source_h5_path_test", type=str,
                        default='/home/SENSETIME/sunxin/2_myrepos/data/blackshark/h5/n2c_lowlight_noised_test.h5')
    parser.add_argument("--source_img_path_test", type=str,
                        default='/home/SENSETIME/sunxin/2_myrepos/data/blackshark/msdct/now')
    parser.add_argument("--test_ep", type=int, default=29)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_images", type=bool, default=True)
    parser.add_argument("--show_images", type=bool, default=True)

    parser.add_argument("--data_root", type=str, default='')
    parser.add_argument("--model_name", type=str, default='')
    return parser.parse_args()


def merge_args_from_train(save_dir, args):
    train_args_path = save_dir + '/train_args.json'
    if os.path.exists(train_args_path):
        print('merging args from:', train_args_path)
        with open(train_args_path, 'r') as f:
            train_args_dict = json.load(f)
        for k, v in train_args_dict.items():
            if k in args.__dict__ and k != 'gpu':
                print('change test_args: [{}] to: {}'.format(k, v))
                args.__dict__[k] = v
    else:
        print("No 'train_args.json', so did not merge args from train_args.")
    return args


def do_test_from_loader(args):
    args = merge_args_from_train(args.model_dir, args)
    model_path = args.model_dir + '/pts/' + str(args.test_ep) + '.pth'
    save_img_dir = args.model_dir + '/save_imgs'
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    state_dict = torch.load(model_path)
    net = choose_model(args.model_name, 'test')
    print('loading:', model_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    net.load_state_dict(state_dict)
    net.eval()
    net.cuda()
    dataset_test = dataset_img2img(args.source_h5_path_test, args.source_h5_path_test.replace('noised', 'clean'))
    loader_test = DataLoader(dataset=dataset_test, num_workers=0, batch_size=1, shuffle=False)
    test_num = len(loader_test)
    ep_psnr_n, ep_ssim_n, ep_psnr_t, ep_ssim_t = 0, 0, 0, 0
    for i, batch in enumerate(loader_test):
        with torch.no_grad():
            input_data = batch[0].cuda()
            batch_labels = batch[1].cuda()
            fw_nlu = net(input_data)
            psnr_n = batch_PSNR(fw_nlu, batch_labels, data_range=1.0)
            ssim_n = ssim(fw_nlu, batch_labels, data_range=1.0, win_size=11).item()
            inference = np.array(fw_nlu.cpu().squeeze(0).permute(1, 2, 0) * MAX).astype('uint8')
            source = np.array(input_data.cpu().squeeze(0).permute(1, 2, 0) * MAX).astype('uint8')
            target = np.array(batch_labels.cpu().squeeze(0).permute(1, 2, 0) * MAX).astype('uint8')

            inference[inference > MAX] = MAX
            inference[inference < 0] = 0

            # if source.shape[2] == 1:
            #     source = source.squeeze()
            #     target = target.squeeze()
            # print(source.shape, target.shape, source.dtype, np.mean(source), psnr_n, ssim_n)

            cv2.imwrite(save_img_dir + '/' + str(i) + '_denoised.bmp', inference)
            cv2.imwrite(save_img_dir + '/' + str(i) + '_noised.bmp', source)
            cv2.imwrite(save_img_dir + '/' + str(i) + '_clean.bmp', target)

            # if args.show_images:
            #     plt.subplot(221)
            #     plt.title('Noised')
            #     plt.imshow(source)
            #     plt.axis('off')
            #     plt.subplot(222)
            #     plt.title('Clean')
            #     plt.imshow(target)
            #     plt.axis('off')
            #     plt.subplot(223)
            #     title = 'Result' + ' PSNR:' + str(np.round(psnr_n, 2)) + ' ' + Path(args.data_root).name
            #     plt.title(title)
            #     plt.imshow(inference)
            #     plt.axis('off')
            #     plt.savefig(save_img_dir + '/' + str(i) + '.png')
            #     plt.show()

            print(i, psnr_n, ssim_n)
            ep_psnr_n += psnr_n
            ep_ssim_n += ssim_n

    ep_psnr_n /= test_num
    ep_ssim_n /= test_num
    print('Final', ep_psnr_n, ep_ssim_n)


def do_test_from_images(args):
    args = merge_args_from_train(args.model_dir, args)
    model_path = args.model_dir + '/pts/' + str(args.test_ep) + '.pth'
    save_img_dir = args.model_dir + '/save_imgs'
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    state_dict = torch.load(model_path)
    net = choose_model(args.model_name, 'test')
    print('loading:', model_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    net.load_state_dict(state_dict)
    net.eval()
    net.cuda()
    random_crop_size = 1024
    paths = [i for i in Path(args.source_img_path_test).glob('*.jpg') if 'mfnr' in i.name]
    for path in paths:
        noised_path = str(path)
        noised = cv2.imread(noised_path)
        noised = cv2.cvtColor(noised, cv2.COLOR_BGR2GRAY)
        h,w = noised.shape
        noised1 = noised[:h // 2, :w // 2]
        noised2 = noised[h // 2:, w // 2:]
        noised3 = noised[h // 2:, :w // 2]
        noised4 = noised[:h // 2, w // 2:]
        noised = noised1
        x = random.randint(1, h // 2 - random_crop_size)
        y = random.randint(1, h // 2 - random_crop_size)
        noised = noised[y:y+random_crop_size, x:x+random_crop_size]

        # clean_path = noised_path.replace('mfnr', 'denoise')
        # clean = cv2.imread(clean_path)
        # clean = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)

        noised = np.array([noised]).transpose((1, 2, 0))
        with torch.no_grad():
            noised = torch.from_numpy(np.float32(noised / MAX)).permute(2, 0, 1)
            noised = noised.unsqueeze(0).cuda()
            print(noised.shape)
            denoised = net(noised)
            denoised = np.array(denoised.cpu().squeeze(0).permute(1, 2, 0) * MAX).astype('uint8')
            denoised[denoised > MAX] = MAX
            denoised[denoised < 0] = 0
        cv2.imwrite(save_img_dir + '/' + path.name[:-4] + '_denoised.jpg', denoised)


if __name__ == "__main__":
    args = get_args()
    search = False
    if search:
        root = 'results_pristine_bsd_kodak_mc/gaussian_50'
        model_dirs = [str(i) for i in Path(root).iterdir() if i.is_dir()]
        for model_dir in model_dirs:
            args.save_dir = model_dir
            do_test_from_loader(args)
    else:
        do_test_from_loader(args)
        # do_test_from_images(args)
