from models.MY import feb_rfb_ab_mish_a_add
from models.BRDNet import BRDNET
from models.ADNet import ADNET
from models.DnCNN import DNCNN
from models.SXNet import SXNet, DNCNN_based
from models.UNet import UNet

from utils.loss import *


def choose_model(model_name, phase):
    if model_name == 'ADNet':
        return ADNET(3, phase)
    elif model_name == 'DnCNN':
        return DNCNN(1, phase)
    elif model_name == 'BRDNet':
        return BRDNET(1, phase)
    elif model_name == 'feb_rfb_ab_mish_a_add':
        return feb_rfb_ab_mish_a_add(3, phase)
    elif 'SXNet' in model_name:
        info = model_name.split('_')
        basic_ch, rfb_ch, asy, tlu, bias = int(info[1]), int(info[2]), False, False, False
        if '_a' in model_name:
            asy = True
        if '_t' in model_name:
            tlu = True
        if '_b' in model_name:
            bias = True
        return SXNet(basic_ch, rfb_ch, 3, phase, asy=asy, bias=bias, tlu=tlu)
    elif 'MKM' == model_name:
        return DNCNN_based(3, 'mkm', phase)
    elif 'RM' == model_name:
        return DNCNN_based(3, 'rm', phase)
    elif 'MKM_RM' == model_name:
        return DNCNN_based(3, 'mkm', phase)
    elif 'Vanilla' == model_name:
        return DNCNN_based(3, 'vanilla', phase)
    elif model_name == 'UNet':
        return UNet(3, phase)

def choose_loss(loss_name):
    if loss_name == 'mse':
        return torch.nn.MSELoss(reduction='mean')
    elif loss_name == 'l1':
        return torch.nn.L1Loss(reduction='mean')
    elif loss_name == 'sl1':
        return torch.nn.SmoothL1Loss(reduction='mean')
    elif loss_name == 'ssim':
        return SSIM_Loss(data_range=1.0, size_average=True, channel=3)
    elif loss_name == 'msssim':
        return MS_SSIM_Loss(data_range=1.0, win_size=3, size_average=True, channel=3)
    elif loss_name == 'hdr':
        return HDRLoss()
    elif loss_name == 'pl':
        return P_Loss()
    elif loss_name == 'me':
        return MyLoss()
