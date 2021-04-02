import cv2
import copy
import torch
from train import choose_model
import numpy as np
from pathlib import Path
import skimage



def show_details():
    for i in Path('./').glob('*.jpg'):
        if i.name == 'n2n.jpg':
            denoised = cv2.imread(str(i))

            denoised_part = copy.deepcopy(denoised)[320:480, 0:80, :]
            denoised_part = cv2.resize(denoised_part, (160, 320))

            show = copy.deepcopy(denoised)
            cv2.rectangle(show, (0, 320 - 2), (80, 480 - 2), color=(0, 0, 255), thickness=2)
            show[160:480, 320:480, :] = denoised_part

            cv2.rectangle(show, (320 - 2, 160 - 2), (480 - 2, 480 - 2), color=(0, 255, 0), thickness=2)

            cv2.imwrite(str(i) + 'show.jpg', show)


def crop_images():
    for i in Path('./').glob('*.jpg'):
        tmp = cv2.imread(str(i))
        tmp = tmp[20:500, 100:580, :]
        cv2.imwrite(str(i), tmp)
    clean = cv2.imread('./0.png')
    clean = clean[20:500, 100:580, :]
    cv2.imwrite('fig8/0.jpg', clean)


def get_denoised_images(dir_num):
    noised = cv2.imread('../dataset/data_tiny/' + str(dir_num) + '/1.png')
    # model_names = ['DNCNN', 'ADNET', 'BRDNET', 'feb_rfb_ab_mish_a_add']
    model_names = ['feb_rfb_ab_mish_a_add']
    for model_name in model_names:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = choose_model(model_name).to(device)
        pth_path = '../results/date-7-4/nuclear_tlu/n2c_DNCNN_bz50_ep50_l1/DNCNN_ep50.pth'
        pth_path = pth_path.replace('DNCNN', model_name)
        model.load_state_dict(torch.load(pth_path))
        image_tensor = torch.from_numpy(noised / 255).unsqueeze(0).permute(0, 3, 1, 2).float().cuda()
        with torch.no_grad():
            batch_inferences = model(image_tensor)
            inference = np.array(batch_inferences.cpu().squeeze(0).permute(1, 2, 0) * 255).astype('uint8')
            cv2.imwrite('/workspace/' + model_name + str(dir_num) + '_tlu.jpg', inference)


def cal_psnr_ssim():
    clean = cv2.imread('fig8/0.jpgshow.jpg')
    clean = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    for i in Path('./').glob('*.jpg'):
        if 'show' in str(i):
            img = cv2.imread(str(i))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            psnr = skimage.measure.compare_psnr(img, clean, 255)
            ssim = skimage.measure.compare_ssim(img, clean, data_range=255)
            print(str(i), psnr, ssim)


if __name__ == '__main__':
    # get_denoised_images(dir_num=2)
    # crop_images()
    # show_details()
    cal_psnr_ssim()

