from pathlib import Path
from utils.add_noise import *
import cv2
import copy
import torch
from train import choose_model
import numpy as np
from pathlib import Path
import skimage
#
# pairs = []
#
# stimulate_clean = cv2.imread('../dataset/BSD300/2092.jpg')
# n1 = add_impulse_noise(stimulate_clean, 25)
# n2 = add_text_noise(stimulate_clean, 25)
# n3 = add_gaussian_noise(stimulate_clean, 25)
# pairs.append((stimulate_clean, n1))
# pairs.append((stimulate_clean, n2))
# pairs.append((stimulate_clean, n3))
#
# real_clean = cv2.imread('../dataset/polyu/Canon5D2_5_160_3200_plug_12_real.JPG')
# n = cv2.imread('../dataset/polyu/Canon5D2_5_160_3200_plug_12_mean.JPG')
# pairs.append((real_clean, n))
#
# for i in range(1, 4):
#     nu_clean1 = cv2.imread('fig9/' + str(i) + '3.jpg')
#     nu_noise1 = cv2.imread('fig9/' + str(i) + '1.jpg')
#     pairs.append((nu_clean1, nu_noise1))
#
# for t, pair in enumerate(pairs):
#     cv2.imwrite(str(t)+'clean.jpg', pair[0][50:300, 20:350, :])
#     cv2.imwrite(str(t)+'noise.jpg', pair[1][50:300, 20:350, :])
#
# def get_denoised_images(noised, type):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     image_tensor = torch.from_numpy(noised / 255).unsqueeze(0).permute(0, 3, 1, 2).float().cuda()
#
#     model = choose_model('BRDNET').to(device)
#     pth_path = '../results/date-7-4/' + type + '/n2c_BRDNET_bz50_ep50_l1/BRDNET_ep50.pth'
#     model.load_state_dict(torch.load(pth_path))
#     with torch.no_grad():
#         batch_inferences = model(image_tensor)
#         with_tlu = np.array(batch_inferences.cpu().squeeze(0).permute(1, 2, 0) * 255).astype('uint8')
#
#     model1 = choose_model('feb_rfb_ab_mish_a_add').to(device)
#     pth_path1 = '../results/date-7-4/' + type + '/n2c_feb_rfb_ab_mish_a_add_bz50_ep50_l1/feb_rfb_ab_mish_a_add_ep50.pth'
#     model1.load_state_dict(torch.load(pth_path1))
#     with torch.no_grad():
#         batch_inferences1 = model1(image_tensor)
#         no_tlu = np.array(batch_inferences1.cpu().squeeze(0).permute(1, 2, 0) * 255).astype('uint8')
#
#     return with_tlu, no_tlu
#
#
# for i, pair in enumerate(pairs):
#     clean = pair[0][50:300, 20:350, :]
#     noise = pair[1][50:300, 20:350, :]
#
#     if i == 2:
#         with_tlu, no_tlu = get_denoised_images(noise, 'gaussian25')
#     elif i == 1:
#         with_tlu, no_tlu = get_denoised_images(noise, 'text25')
#     elif i == 0:
#         with_tlu, no_tlu = get_denoised_images(noise, 'implus25')
#     elif i == (len(pairs) - 1):
#         with_tlu, no_tlu = get_denoised_images(noise, 'polyu')
#     else:
#         with_tlu, no_tlu = get_denoised_images(noise, 'nuclear')
#
#     cv2.imwrite('noise' + str(i) + '.jpg', noise)
#     cv2.imwrite('with_tlu' + str(i) + '.jpg', with_tlu)
#     cv2.imwrite('no_tlu' + str(i) + '.jpg', no_tlu)

for i in range(1, 4):
    tmp = cv2.imread('fig9/' + str(i) + '2.jpg')

    cv2.imwrite(str(i)+'ss.jpg', tmp[50:300, 20:350, :])
