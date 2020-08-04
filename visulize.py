import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path
import json
import xlwt
from train import choose_model
import cv2

from train import dataset_img2img
from torch.utils.data import DataLoader


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


def qualitative_results(model_name, model_path, data_path, vis_type, save_dir):
    assert model_name in model_path
    data_path = data_path.replace('_tlu', '')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = choose_model(model_name).to(device)
    model.load_state_dict(torch.load(model_path))

    if '.avi' in data_path:
        cap = cv2.VideoCapture('demo/demo.avi')
        total_frame_num_VIDEO = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = None
        if vis_type == 'write':
            frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            dstSize = (int(frameWidth), int(frameHeight))
            out = cv2.VideoWriter(save_dir + "/demo_denoised.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, dstSize)

        for frame_idx in range(total_frame_num_VIDEO):
            ret, frame = cap.read()
            print('processing frame idx', frame_idx)
            with torch.no_grad():
                frame_tensor = torch.from_numpy(frame / 255).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
                batch_inferences = model(frame_tensor)
                inference = np.array(batch_inferences.cpu().squeeze(0).permute(1, 2, 0) * 255).astype('uint8')
                if vis_type == 'write':
                    out.write(inference)
                else:
                    cv2.imshow('inference', inference)
                    cv2.waitKey(1)

    elif '.h5' in data_path:
        h5_path_label = data_path.replace('source', 'target')
        assert len(data_path) == len(h5_path_label)
        dataset = dataset_img2img(data_path, h5_path_label)
        loader = DataLoader(dataset=dataset, num_workers=0, batch_size=1, shuffle=False)
        for i, batch in enumerate(loader):
            batch_datas = batch[0].to(device=device)
            batch_labels = batch[1].to(device=device)
            with torch.no_grad():
                batch_inferences = model(batch_datas)
                inference = np.array(batch_inferences.cpu().squeeze(0).permute(1, 2, 0) * 255).astype('uint8')
                source = np.array(batch_datas.cpu().squeeze(0).permute(1, 2, 0) * 255).astype('uint8')
                target = np.array(batch_labels.cpu().squeeze(0).permute(1, 2, 0) * 255).astype('uint8')
                result = cv2.vconcat([source, target, inference])
                if vis_type == 'write':
                    print(save_dir + '/' + str(i) + '.jpg')
                    cv2.imwrite(save_dir + '/i' + str(i) + '.jpg', result)
                else:
                    result = Image.fromarray(result.astype('uint8'))
                    plt.imshow(result)
                    plt.show()

    elif '.' not in data_path:
        images = [i for i in Path(data_path).iterdir() if i.is_file()]
        for image_path in images:
            image = cv2.imread(str(image_path))
            image_tensor = torch.from_numpy(image / 255).unsqueeze(0).permute(0, 3, 1, 2)
            batch_inferences = model(image_tensor)
            inference = np.array(batch_inferences.cpu().squeeze(0).permute(1, 2, 0) * 255).astype('uint8')
            result = cv2.hconcat([image, inference])
            if vis_type == 'write':
                cv2.imwrite(save_dir + '/' + image_path.name + '.jpg', result)
            else:
                result = Image.fromarray(result.astype('uint8'))
                plt.imshow(result)
                plt.show()


def quantative_results(results_dir):
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('sheet', cell_overwrite_ok=True)
    sheet.write(0, 0, 'dataset')
    sheet.write(0, 1, 'model')
    sheet.write(0, 2, 'loss')
    sheet.write(0, 3, 'psnr')
    sheet.write(0, 4, 'ssim')

    root = 'results/' + results_dir
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

    book.save('results/' + results_dir + '.xls')


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


if __name__ == '__main__':
    results_dir = 'date-7-4'
    noise = 'implus50'
    model = 'feb_rfb_ab_mish_a_add'
    loss = 'l1'
    function = 0

    if function == 0:
        quantative_results(results_dir)
    elif function == 1:
        logfile_path = 'results/' + results_dir + '/' + noise + '/n2c_' + model + '_bz50_ep50_' + loss + '/logs.json'
        training_curves(logfile_path)
    elif function == 2:
        model_path = 'results/' + results_dir + '/' + noise + '/n2c_' + model + '_bz50_ep50_l1/' + model + '_ep50.pth'
        data_path = 'h5/n2c_' + noise + '_source_test.h5'
        qualitative_results(model, model_path, data_path, 'write', 'tmp')

