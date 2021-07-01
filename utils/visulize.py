import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import os
import json
import h5py
import operator
import random
from argparse import ArgumentParser
import torch.optim as optim
import torch
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path
import json
import xlwt
from tqdm import tqdm
from train_h5 import choose_model
import cv2
import time


def vis_a_image(input, type, channel, ax, decode):
    if type == 'pil':
        plt.subplot(ax)
        plt.imshow(input)
    elif type == 'np':
        pass
    elif type == 'tensor' and channel == 4:
        input = input.squeeze(0)  # C0 H1 W2
        input = input.permute(1, 2, 0)
        input = np.array(input)
        if decode == True:
            input *= 255
        input = Image.fromarray(input.astype('uint8'))
        plt.subplot(ax)
        plt.imshow(input)


def inference_from_cv2(input_mat, model):
    assert input_mat.shape[2] <= 3  # HWC
    input_mat = torch.from_numpy(np.float32(input_mat / 255)).permute(2, 0, 1)  # CHW,normalized,torch tensor
    batch_tensor = input_mat.unsqueeze(0).cuda()  # NCHW,cuda tensor
    with torch.no_grad():
        a = time.clock()
        batch_inferences = model(batch_tensor)
        fps = np.round(1 / (time.clock() - a), 3)
    inference = np.array(batch_inferences.cpu().squeeze(0).permute(1, 2, 0) * 255).astype('uint8')
    return inference, fps


def qualitative_results(model_name, model_path, data_path, save_dir):
    data_path = data_path.replace('_tlu', '')  # 只推理不含tlu的模型
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    model = choose_model(model_name, 'test')
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 非常重要
    model.cuda()
    if Path(data_path).is_file() and '.avi' in data_path:
        cap = cv2.VideoCapture(data_path)
        out = cv2.VideoWriter(save_dir + "/tmp.flv",
                              cv2.VideoWriter_fourcc('F', 'L', 'V', '1'),
                              cap.get(cv2.CAP_PROP_FPS),
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            _, frame = cap.read()
            inference, fps = inference_from_cv2(frame, model)
            # print('FPS:', fps, frame.dtype)
            # plt.imshow(cv2.cvtColor(inference, cv2.COLOR_BGR2RGB))
            # plt.show()
            out.write(inference)
    elif Path(data_path).is_file() and '_test.h5' in data_path:
        h5 = h5py.File(data_path, 'r')
        keys = list(h5.keys())
        for key in tqdm(keys):
            noised = np.array(h5[key])
            inference, fps = inference_from_cv2(noised, model)
            print('FPS:', fps, noised.dtype)
            plt.imshow(cv2.cvtColor(inference, cv2.COLOR_BGR2RGB))
            plt.show()
    else:
        assert Path(data_path).is_dir()
        images = [i for i in Path(data_path).glob('*.jpg')]
        for image_path in tqdm(images):
            image = cv2.imread(str(image_path))
            # image = Image.open(str(image_path))
            # image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            inference, fps = inference_from_cv2(image, model)
            print('FPS:', fps, image.dtype)
            cv2.imwrite(save_dir + '/' + image_path.name, inference)
            # plt.imshow(cv2.cvtColor(inference, cv2.COLOR_BGR2RGB))
            # plt.show()


def quantative_results(results_dir):
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('sheet', cell_overwrite_ok=True)
    sheet.write(0, 0, 'dataset')
    sheet.write(0, 1, 'model')
    sheet.write(0, 2, 'loss')
    sheet.write(0, 3, 'psnr')
    sheet.write(0, 4, 'ssim')

    root = 'results_nuclear_real_denoise/' + results_dir
    dataset_dirs = [i for i in Path(root).iterdir() if i.is_dir()]
    row = 1
    for dataset_dir in dataset_dirs:
        model_dirs = [i for i in Path(dataset_dir).iterdir() if i.is_dir()]
        for dir in model_dirs:
            info = str(dir).split('/')
            json_file = str(dir) + '/logs.json'
            if os.path.exists(json_file):
                with open(json_file) as f:
                    dicts = json.load(f)
                select_dict = {}
                lowest_loss = 100
                for dict in dicts:
                    if dict['loss'] < lowest_loss:
                        select_dict = dict
                        lowest_loss = dict['loss']

                sheet.write(row, 0, info[2])
                sheet.write(row, 1, info[3])
                sheet.write(row, 2, select_dict['loss'])
                sheet.write(row, 3, select_dict['psnr'])
                sheet.write(row, 4, select_dict['ssim'])
                row += 1

    book.save('results_nuclear_real_denoise/' + results_dir + '.xls')


def training_curves(logfile_path):
    with open(logfile_path) as f:
        l1 = json.load(f)

    logfile_path2 = logfile_path.replace('l1', 'mse')
    with open(logfile_path2) as f2:
        l2 = json.load(f2)

    epochs = [i for i in range(50)]
    losses_l1 = [i['loss'] * 1e4 for i in l1]
    psnrs_l1 = [i['psnr'] for i in l1]
    ssims_l1 = [i['ssim'] for i in l1]

    psnrs_l2 = [i['psnr'] for i in l2]
    ssims_l2 = [i['ssim'] for i in l2]
    losses_l2 = [i['loss'] * 1e5 for i in l2]

    def show_psnr():
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(epochs, psnrs_l1, '-', label='L1')
        ax.plot(epochs, psnrs_l2, '-r', label='L2')

        ax.legend(loc=0)
        ax.grid()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("PSNR (dB)")
        ax.set_ylim(25, 35)

        plt.savefig("psnr.png")
        plt.show()

    def show_ssim():
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(epochs, ssims_l1, '-', label='L1')
        ax.plot(epochs, ssims_l2, '-r', label='L2')

        ax.legend(loc=0)
        ax.grid()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("SSIM")
        ax.set_ylim(0.7, 1)

        plt.savefig("ssim.png")
        plt.show()

    def show_loss():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel("Epoch")

        lns1 = ax.plot(epochs, losses_l1, '-', label='L1 loss')

        ax.set_ylim(130, 380)
        plt.yticks([])
        ax2 = ax.twinx()
        lns2 = ax2.plot(epochs, losses_l2, '-r', label='L2 loss')

        ax2.set_ylim(45, 200)

        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=0)

        plt.yticks([])

        plt.savefig("loss.png")
        plt.show()

    show_psnr()
    show_ssim()
    show_loss()


def read_logs(save_dirs_root, val_name):
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('sheet', cell_overwrite_ok=True)
    sheet.write(0, 0, 'noise_name')
    sheet.write(0, 1, 'model')
    sheet.write(0, 2, 'loss')
    sheet.write(0, 3, 'psnr')
    sheet.write(0, 4, 'ssim')
    row = 1
    save_dirs = [i for i in Path(save_dirs_root).iterdir() if i.is_dir()]
    for save_dir in save_dirs:
        info = str(save_dir).split('/')
        json_file = str(save_dir) + '/' + val_name + '_logs.json'
        if os.path.exists(json_file):
            with open(json_file) as f:
                dicts = json.load(f)
            select_dict = {}
            lowest_loss = 100
            for dict in dicts:
                if dict['loss'] < lowest_loss:
                    select_dict = dict
                    lowest_loss = dict['loss']
            sheet.write(row, 0, info[-2])
            sheet.write(row, 1, info[-1])
            sheet.write(row, 2, select_dict['loss'])
            if 'psnr_t' in select_dict:
                assert 'ssim_t' in select_dict
                sheet.write(row, 3, select_dict['psnr_t'])
                sheet.write(row, 4, select_dict['ssim_t'])
            else:
                sheet.write(row, 3, select_dict['psnr_n'])
                sheet.write(row, 4, select_dict['ssim_n'])
            row += 1
            print(json_file, select_dict)
        else:
            print('No json file!', json_file)
    book.save(save_dirs_root + '/' + Path(save_dirs_root).name + '.xls')


if __name__ == '__main__':
    search = False

    if search:
        results_dir = 'asy_nobias'
        noise = 'multi_10'
        model = 'feb_rfb_ab_mish_a_add'
        loss = 'l1'
        function = 0
        if function == 0:
            quantative_results(results_dir)
        elif function == 1:
            logfile_path = 'results_nuclear_real_denoise/' + results_dir + '/' + noise + '/n2c_' + model + '_bz64_ep50_' + loss + '/logs.json'
            training_curves(logfile_path)
        elif function == 2:
            model_path = 'results_nuclear_real_denoise/' + results_dir + '/' + noise + '/n2c_' + model + '_bz64_ep50_l1/' + model + '_ep50.pth'
            data_path = 'h5/n2c_' + noise + '_source_test.h5'
            qualitative_results(model, model_path, data_path, 'write', 'tmp')
    else:
        # vis_type = 'write'

        # model_name = 'feb_rfb_ab_mish_a_add'
        # model_path = 'results_nuclear_real_denoise/multi_10/feb_rfb_ab_mish_a_add-bz64_ep50_l1/feb_rfb_ab_mish_a_add-bz64_ep50_l1_ep32.pth'

        # model_name = 'SXNet_8_16_a'
        # model_path = 'results_nuclear_real_denoise/nuclear/SXNet_8_16_a-bz64_ep50_l1/SXNet_8_16_a-bz64_ep50_l1_ep50.pth'

        # data_path = '/home/ipsg/code/sx/datasets/videos/nuclear_real_cropped.avi'
        # save_dir = 'tmp'

        # data_path = '/home/ipsg/code/sx/datasets/nuclear_cold_tiny/JPEGImages_multi_10'
        # save_dir = data_path + '_denoised'

        # data_path = '/home/ipsg/code/sx/datasets/nuclear_cold_tiny/n2c_multi_10_noised_test.h5'
        # save_dir = None

        read_logs('results/BSD300/gaussian_50/bz64', 'McMaster')
