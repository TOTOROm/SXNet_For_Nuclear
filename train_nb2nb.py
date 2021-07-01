import os

os.environ['CUDA_VISBILE_DEVICES'] = '1'

import torch

torch.cuda.set_device(1)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from importlib import import_module
from torch.utils.data import Dataset
import sys
import random
import glob
import time
import argparse
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import cv2
import json
from argparse import ArgumentParser
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.init as init
from torch.optim import lr_scheduler
from torch.autograd import Variable
from ptflops import get_model_complexity_info
import time
from skimage.measure.simple_metrics import compare_psnr

from utils.data import dataset_single_h5
from models.UNet_Nb2Nb import UNet_Nb2Nb
from models.DnCNN import DnCNN
from utils.data import image2patches_random2


class ImagePairDataset(Dataset):
    def __init__(self, data_root, size=256):
        super(ImagePairDataset, self).__init__()
        self.size = size
        self.paths = [i for i in Path(data_root).rglob('*.png') if '0.png' not in i.name]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = cv2.imread(str(path))
        h, w, c = image.shape
        image = image[:h // 2 * 2, :w // 2 * 2, :]

        image2 = cv2.imread(str(path.parent) + '/0.png')
        assert image.shape == image2.shape
        image2 = image2[:h // 2 * 2, :w // 2 * 2, :]

        if self.size > 0:
            h, w, c = image.shape
            rand_tl_x = random.randint(0, w - self.size)
            rand_tl_y = random.randint(0, h - self.size)
            br_y = rand_tl_y + self.size
            br_x = rand_tl_x + self.size
            img_crop = image[rand_tl_y: br_y, rand_tl_x:br_x]

            img_crop2 = image2[rand_tl_y: br_y, rand_tl_x:br_x]
        else:
            img_crop = image

            img_crop2 = image2
        img_crop = torch.from_numpy(img_crop).permute(2, 0, 1).float()  # HWC to CHW
        img_crop2 = torch.from_numpy(img_crop2).permute(2, 0, 1).float()  # HWC to CHW
        return img_crop, img_crop2


class masksamplingv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(1, 2)

    def forward(self, input, patten):
        _, _, w, h = input.size()
        assert w % 2 == 0
        assert h % 2 == 0
        if patten == 0:
            output1 = self.pool(input)
            output2 = self.pool(input[:, :, 1:, :])
            return output2, output1
        elif patten == 1:
            output1 = self.pool(input)
            output2 = self.pool(input[:, :, 1:, :])
            return output2, output1
        elif patten == 2:
            output1 = self.pool(input)
            output3 = self.pool(input[:, :, :, 1:])
            return output1, output3
        elif patten == 3:
            output1 = self.pool(input)
            output3 = self.pool(input[:, :, :, 1:])
            return output3, output1
        elif patten == 4:
            output2 = self.pool(input[:, :, 1:, :])
            output4 = self.pool(input[:, :, 1:, 1:])
            return output4, output2
        elif patten == 5:
            output2 = self.pool(input[:, :, 1:, :])
            output4 = self.pool(input[:, :, 1:, 1:])
            return output2, output4
        elif patten == 6:
            output3 = self.pool(input[:, :, :, 1:])
            output4 = self.pool(input[:, :, 1:, 1:])
            return output3, output4
        elif patten == 7:
            output3 = self.pool(input[:, :, :, 1:])
            output4 = self.pool(input[:, :, 1:, 1:])
            return output4, output3
        else:
            raise Exception("no implemented patten")


def generate_mask_pair(tensorNCHW):
    xy_pairs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    grid1 = torch.zeros(tensorNCHW.size())
    grid2 = torch.zeros(tensorNCHW.size())
    # print('grid shapes:', grid1.shape, grid2.shape)
    _, _, h, w = grid1.shape
    for y in range(0, h - 2, 2):
        for x in range(0, w - 2, 2):
            xy_id1, xy_id2 = 0, 0
            while xy_pairs[xy_id1] == xy_pairs[xy_id2]:
                xy_id1 = random.randint(0, 3)  # [0,3]
                xy_id2 = random.randint(0, 3)  # [0,3]
            xy_pair1 = xy_pairs[xy_id1]
            xy_pair2 = xy_pairs[xy_id2]
            # print('axis of 1 in the pairs', xy_pair1, xy_pair2)
            red_y = y + xy_pair1[0]
            red_x = x + xy_pair1[1]
            blue_y = y + xy_pair2[0]
            blue_x = x + xy_pair2[1]
            grid1[:, :, red_y, red_x] = 1
            grid2[:, :, blue_y, blue_x] = 1
    return grid1, grid2


def generate_subimages(img_tensor, mask):
    import torch.nn.functional as F
    out = F.max_pool2d(img_tensor * mask, kernel_size=2, stride=2)
    return out


def train_real(config):
    network = UNet_Nb2Nb(c_in=3, c_out=3, post_processing=True)
    # network = UNet_N2N()
    # network = DnCNN(3)

    trainset = dataset_single_h5(config['data_root'])
    loader_train = DataLoader(dataset=trainset, batch_size=config['batch_size'], shuffle=True)

    init_lr = 1e-4
    network.cuda()

    optimizer = optim.Adam(network.parameters(), lr=init_lr)
    n_epoch = config['epoches']
    for epoch in range(1, n_epoch + 1):
        network.train()
        if epoch % 20 == 0:
            init_lr /= 2
        for param_group in optimizer.param_groups:
            param_group["lr"] = init_lr

        for iteration, noisy in enumerate(loader_train):
            # preparing synthetic noisy images
            noisy = noisy / 255.0
            # print(noisy.shape)
            # generating a sub-image pair
            mask1, mask2 = generate_mask_pair(noisy)
            noisy_sub1 = generate_subimages(noisy, mask1).cuda()
            noisy_sub2 = generate_subimages(noisy, mask2).cuda()
            # preparing for the regularization term
            with torch.no_grad():
                noisy_denoised = network(noisy.cuda())
            noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1.cuda())
            noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2.cuda())
            # calculating the loss
            noisy_output = network(noisy_sub1)
            noisy_target = noisy_sub2
            # Lambda = epoch / n_epoch * ratio
            Lambda = 1 + (epoch / n_epoch)
            diff = noisy_output - noisy_target
            exp_diff = noisy_sub1_denoised - noisy_sub2_denoised
            loss1 = torch.mean(diff ** 2)
            loss2 = Lambda * torch.mean((diff - exp_diff) ** 2)
            loss_all = loss1 + loss2

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            if iteration % 10 == 0:
                print("[epoch %d][%d/%d] loss: %.4f loss1: %.4f loss2: %.4f gamma: %.2f lr: %8f" %
                      (epoch, iteration, len(loader_train),
                       loss_all.item(), loss1.item(), loss2.item(),
                       Lambda, optimizer.param_groups[0]['lr']))

        network.eval()
        # if epoch % 2 == 0:
        torch.save(network.state_dict(), os.path.join(config['save_dir'], str(epoch) + '.pth'))


def eval_real(config):
    test_pth_name = '72.pth'

    save_root = str(Path(config['data_root']).parent)
    model_name = Path(config['save_dir']).name
    save_img_dir = save_root + '/' + model_name + '_eval'
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)

    network = UNet_Nb2Nb(3, 3, post_processing=True)
    # network = UNet_N2N()
    # network = DnCNN(3)

    state_dicts = torch.load(os.path.join(config['save_dir'], test_pth_name))
    network.load_state_dict(state_dicts)
    network.cuda()
    network.eval()
    test_num = 0
    for path in Path(save_root).glob('*.jpg'):
        source = cv2.imread(str(path))
        patches = image2patches_random2(source, 512, 4)
        for patch in patches:
            print(test_num, patch.shape, str(path))
            cv2.imwrite(save_img_dir + '/' + str(test_num) + "_ori.jpg", patch,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            noisy = torch.from_numpy(patch / 255.0).permute(2, 0, 1).float().unsqueeze(0)  # NCHW
            with torch.no_grad():
                denoised = network(noisy.cuda())

            # mask1, mask2 = generate_mask_pair(noisy)
            # noisy_sub1 = generate_subimages(noisy, mask1).cuda()
            # noisy_sub2 = generate_subimages(noisy, mask2).cuda()
            # vis_IN_OUT_source_target(noisy, denoised, noisy_sub1, noisy_sub2)
            from utils.data import vis_IN_OUT
            vis_IN_OUT(noisy, denoised)

            denoised = denoised.cpu().detach().squeeze(0).permute(1, 2, 0)  # CHW to HWC
            denoised = np.array(torch.clamp(denoised * 255.0, 0, 255)).astype('uint8')
            cv2.imwrite(save_img_dir + '/' + str(test_num) + "_dn.jpg", denoised,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            test_num += 1


def train_real_with_gt(config):
    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'])
    valset = ImagePairDataset(config['data_root'], size=-1)

    network = UNet_Nb2Nb(3, 3, post_processing=True)
    # network = UNet_N2N()

    network.cuda()
    optimizer = define_optim('adam', network.parameters(), 1e-3, 0)
    n_epoch = config['epoches']
    # ratio = 1
    for epoch in range(1, n_epoch + 1):
        trainset = ImagePairDataset(config['data_root'], size=config['crop_size'])
        loader_train = DataLoader(dataset=trainset, batch_size=config['batch_size'], shuffle=True)
        network.train()
        for iteration, pair in enumerate(loader_train):
            noisy = pair[0] / 255.0
            clean = pair[1] / 255.0

            mask1, mask2 = generate_mask_pair(noisy)
            noisy_sub1 = generate_subimages(noisy, mask1).cuda()
            noisy_sub2 = generate_subimages(noisy, mask2).cuda()
            with torch.no_grad():
                noisy_denoised = network(noisy.cuda())
            noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1.cuda())
            noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2.cuda())
            noisy_output = network(noisy_sub1)
            noisy_target = noisy_sub2
            # Lambda = epoch / n_epoch * ratio
            Lambda = 1 + (epoch / n_epoch)
            diff = noisy_output - noisy_target
            exp_diff = noisy_sub1_denoised - noisy_sub2_denoised
            loss1 = torch.mean(diff ** 2)
            loss2 = Lambda * torch.mean((diff - exp_diff) ** 2)
            loss_all = loss1 + loss2

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            # vis_IN_OUT_source_target(noisy, clean, noisy_sub1, noisy_sub2)

            if iteration % 10 == 0:
                print("[epoch %d][%d/%d] loss: %.4f loss1: %.4f loss2: %.4f gamma: %.2f lr: %8f" %
                      (epoch, iteration + 1, len(loader_train),
                       loss_all.item(), loss1.item(), loss2.item(),
                       Lambda, optimizer.param_groups[0]['lr']))

        network.eval()
        psnr_val = 0
        for k in range(len(valset)):
            clean = torch.unsqueeze(valset[k][1] / 255.0, 0)
            noisy = torch.unsqueeze(valset[k][0] / 255.0, 0)
            with torch.no_grad():
                denoised = network(noisy.cuda())
            psnr_val += batch_PSNR(clean.cuda(), denoised.cuda(), 1.)

            # vis_IN_OUT_GT(noisy, denoised, clean)

        psnr_val /= len(valset)
        print("##############################[epoch %d] PSNR_val: %.4f" % (epoch, psnr_val))
        if epoch % 10 == 0:
            torch.save(network.state_dict(), os.path.join(config['save_dir'], str(epoch) + '.pth'))


def eval_real_with_gt(config):
    state_dicts = torch.load('UNet_N2N/nuclear/100.pth')
    network = UNet_Nb2Nb(3, 3, post_processing=True)
    # network = UNet_N2N()
    # network = DnCNN(3)
    network.load_state_dict(state_dicts)
    network.cuda()
    valset = ImagePairDataset(config['data_root'], size=-1)
    network.eval()
    psnr_val = 0
    for k in range(len(valset)):
        noised = np.array(valset[k][0].permute(1, 2, 0)).astype('uint8')
        cv2.imwrite('tmp/' + str(k) + "ori.jpg", noised, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # GT = np.array(valset[k][1].permute(1, 2, 0)).astype('uint8')
        # cv2.imwrite('tmp/' + str(k) + "gt.jpg", GT, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        noisy = torch.unsqueeze(valset[k][0] / 255.0, 0)
        clean = torch.unsqueeze(valset[k][1] / 255.0, 0)

        with torch.no_grad():
            denoised = network(noisy.cuda())
            psnr = batch_PSNR(clean.cuda(), denoised.cuda(), 1.)
            print('PSNR:{}'.format(psnr))
            psnr_val += psnr
            denoised = denoised.cpu().detach().squeeze(0)
            denoised = denoised.permute(1, 2, 0) * 255  # CHW to HWC
            denoised = torch.clamp(denoised, 0, 255)
            denoised = np.array(denoised).astype('uint8')
            cv2.imwrite('tmp/' + str(k) + "dn.jpg", denoised, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    print('Average PSNR:{}'.format(psnr_val / len(valset)))


if __name__ == '__main__':
    config = {
        # 'data_root': "/home/SENSETIME/sunxin/data/front/nb2nb/n2c_nb2nb_train.h5",
        # 'save_dir': "/home/SENSETIME/sunxin/data/front/nb2nb_model/",
        'data_root': "/home/ipsg/code/sx/datasets/front/n2c_nb2nb_train.h5",
        'save_dir': "/home/ipsg/code/sx/datasets/front/nb2nb/",
        'epoches': 100,
        'batch_size': 4,
    }
    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'])

    # train_synthetic(config)
    # eval_synthetic(config)

    # train_real(config)
    eval_real(config)

    # train_real_with_gt(config)
    # eval_real_with_gt(config)
