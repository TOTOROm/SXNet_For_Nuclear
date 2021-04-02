from torchvision.models import vgg19
import torch
import torch.nn as nn
from utils.metric import MS_SSIM, SSIM


class VGG_FeatureExtractor(nn.Module):
    def __init__(self, device):
        super(VGG_FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True).to(device)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, x):
        out = self.feature_extractor(x)
        return out


class P_Loss(nn.Module):
    def __init__(self, criterion='mse', device='cuda'):
        super(P_Loss, self).__init__()
        self.feature_extractor = VGG_FeatureExtractor(device)
        self.device = device
        if criterion == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif criterion == 'mse':
            self.criterion = torch.nn.MSELoss()

    def forward(self, pred, gt):
        pred_feature = self.feature_extractor(pred).to(self.device)
        gt_feature = self.feature_extractor(gt).to(self.device)
        loss = self.criterion(pred_feature, gt_feature)
        return loss


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, pred, gt):
        computeL1 = torch.nn.L1Loss()
        l1 = computeL1(pred, gt)
        computeP = P_Loss()
        p = computeP(pred, gt)
        loss = l1 * 0.8 + p * 0.2
        return loss


class MS_SSIM_Loss(MS_SSIM):
    def forward(self, img1, img2):
        return 100 * (1 - super(MS_SSIM_Loss, self).forward(img1, img2))


class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 100 * (1 - super(SSIM_Loss, self).forward(img1, img2))


class HDRLoss(nn.Module):
    def __init__(self, eps=0.01):
        super(HDRLoss, self).__init__()
        self._eps = eps

    def forward(self, denoised, target):
        loss = ((denoised - target) ** 2) / (denoised + self._eps) ** 2
        return torch.mean(loss.view(-1))
