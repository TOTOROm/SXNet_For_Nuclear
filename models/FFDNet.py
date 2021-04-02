import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import cv2
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from skimage.measure.simple_metrics import compare_psnr
from skimage.util import random_noise


def is_image_gray(image):
    """
    :param image: cv2
    """
    # a[..., 0] == a.T[0].T
    return not (len(image.shape) == 3 and not (
                np.allclose(image[..., 0], image[..., 1]) and np.allclose(image[..., 2], image[..., 1])))


def downsample(x):
    """
    :param x: (C, H, W)
    :param noise_sigma: (C, H/2, W/2)
    :return: (4, C, H/2, W/2)
    """
    # x = x[:, :, :x.shape[2] // 2 * 2, :x.shape[3] // 2 * 2]
    N, C, W, H = x.size()
    idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]

    Cout = 4 * C
    Wout = W // 2
    Hout = H // 2

    if 'cuda' in x.type():
        down_features = torch.cuda.FloatTensor(N, Cout, Wout, Hout).fill_(0)
    else:
        down_features = torch.FloatTensor(N, Cout, Wout, Hout).fill_(0)

    for idx in range(4):
        down_features[:, idx:Cout:4, :, :] = x[:, :, idxL[idx][0]::2, idxL[idx][1]::2]

    return down_features


def upsample(x):
    """
    :param x: (n, C, W, H)
    :return: (n, C/4, W*2, H*2)
    """
    N, Cin, Win, Hin = x.size()
    idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]

    Cout = Cin // 4
    Wout = Win * 2
    Hout = Hin * 2

    up_feature = torch.zeros((N, Cout, Wout, Hout)).type(x.type())
    for idx in range(4):
        up_feature[:, :, idxL[idx][0]::2, idxL[idx][1]::2] = x[:, idx:Cin:4, :, :]

    return up_feature


class FFDNet(nn.Module):

    def __init__(self, is_gray):
        super(FFDNet, self).__init__()

        if is_gray:
            self.num_conv_layers = 15  # all layers number
            self.downsampled_channels = 5  # Conv_Relu in
            self.num_feature_maps = 64  # Conv_Bn_Relu in
            self.output_features = 4  # Conv out
        else:
            self.num_conv_layers = 12
            self.downsampled_channels = 15
            self.num_feature_maps = 96
            self.output_features = 12

        self.kernel_size = 3
        self.padding = 1

        layers = []
        # Conv + Relu
        layers.append(nn.Conv2d(in_channels=self.downsampled_channels, out_channels=self.num_feature_maps, \
                                kernel_size=self.kernel_size, padding=self.padding, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Conv + BN + Relu
        for _ in range(self.num_conv_layers - 2):
            layers.append(nn.Conv2d(in_channels=self.num_feature_maps, out_channels=self.num_feature_maps, \
                                    kernel_size=self.kernel_size, padding=self.padding, bias=False))
            layers.append(nn.BatchNorm2d(self.num_feature_maps))
            layers.append(nn.ReLU(inplace=True))

        # Conv
        layers.append(nn.Conv2d(in_channels=self.num_feature_maps, out_channels=self.output_features, \
                                kernel_size=self.kernel_size, padding=self.padding, bias=False))

        self.intermediate_dncnn = nn.Sequential(*layers)

    # def forward(self, x, noise_sigma):
    #     noise_map = noise_sigma.view(x.shape[0], 1, 1, 1).repeat(1, x.shape[1], x.shape[2] // 2, x.shape[3] // 2)
    #
    #     x_up = utils.downsample(x.data)  # 4 * C * H/2 * W/2
    #     x_cat = torch.cat((noise_map.data, x_up), 1)  # 4 * (C + 1) * H/2 * W/2
    #     x_cat = Variable(x_cat)
    #
    #     h_dncnn = self.intermediate_dncnn(x_cat)
    #     y_pred = utils.upsample(h_dncnn)
    #     return y_pred

    def forward(self, x):
        noise_sigma = torch.FloatTensor(np.array([30 for _ in range(x.shape[0])])).cuda()
        noise_map = noise_sigma.view(x.shape[0], 1, 1, 1).repeat(1, x.shape[1], x.shape[2] // 2, x.shape[3] // 2)
        x_up = downsample(x.data)  # 4 * C * H/2 * W/2
        x_cat = torch.cat((noise_map.data, x_up), 1)  # 4 * (C + 1) * H/2 * W/2
        x_cat = Variable(x_cat)
        h_dncnn = self.intermediate_dncnn(x_cat)
        y_pred = upsample(h_dncnn)
        return y_pred



if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    import time

    with torch.no_grad():
        net = FFDNet(is_gray=False).cuda()

        f, p = get_model_complexity_info(net, (3, 480, 640), as_strings=True, print_per_layer_stat=False, verbose=False)
        print('FLOPs:', f, 'Parms:', p)

        x = torch.randn(1, 3, 480, 640).cuda()
        s = time.clock()
        y = net(x)
        print(y.shape, 1/(time.clock()-s))
