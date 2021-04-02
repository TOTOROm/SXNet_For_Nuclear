import torch
import torch.nn as nn


# 只有最后一层bias=True，其余为False
class ADNET(nn.Module):
    def __init__(self, in_c, phase):
        super(ADNET, self).__init__()
        self.phase = phase
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=64, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, bias=False, dilation=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, bias=False, dilation=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_6 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_7 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_8 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, bias=False, dilation=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_10 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_11 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_12 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, bias=False, dilation=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_13 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_14 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_15 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.conv1_16 = nn.Conv2d(in_channels=64, out_channels=in_c, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=in_c * 2, out_channels=in_c, kernel_size=1, stride=1, padding=0, groups=1,
                               bias=True)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1)
        x1 = self.conv1_5(x1)
        x1 = self.conv1_6(x1)
        x1 = self.conv1_7(x1)
        x1 = self.conv1_8(x1)
        x1 = self.conv1_9(x1)
        x1 = self.conv1_10(x1)
        x1 = self.conv1_11(x1)
        x1 = self.conv1_12(x1)
        x1 = self.conv1_13(x1)
        x1 = self.conv1_14(x1)
        x1 = self.conv1_15(x1)
        x1 = self.conv1_16(x1)
        out = torch.cat([x, x1], 1)
        out = self.Tanh(out)
        out = self.conv3(out)
        out = out * x1
        out2 = x - out
        if self.phase == 'train':
            return out2, None
        else:
            return out2


if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    import time

    with torch.no_grad():
        net = ADNET(3, 'test').cuda()

        f, p = get_model_complexity_info(net, (3, 480, 640), as_strings=True, print_per_layer_stat=False, verbose=False)
        print('FLOPs:', f, 'Parms:', p)

        x = torch.randn(1, 3, 480, 640).cuda()
        s = time.clock()
        y = net(x)
        print(y.shape, 1 / (time.clock() - s))
