import torch
import torch.nn as nn


# 只有第一层bias=True，其余为False
class DNCNN(nn.Module):
    def __init__(self, in_c, phase):
        super(DNCNN, self).__init__()
        self.phase = phase
        self.conv_1_1 = nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=64, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        self.conv_1_2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_5 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_6 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_7 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_8 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_9 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_10 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_11 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_12 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_13 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_14 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_15 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_16 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm2d(64), nn.ReLU())

        self.conv_1_17 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=in_c, kernel_size=3, padding=1, bias=False))

    def forward(self, x):
        x1 = self.conv_1_1(x)
        x1 = self.conv_1_2(x1)
        x1 = self.conv_1_3(x1)
        x1 = self.conv_1_4(x1)
        x1 = self.conv_1_5(x1)
        x1 = self.conv_1_6(x1)
        x1 = self.conv_1_7(x1)
        x1 = self.conv_1_8(x1)
        x1 = self.conv_1_9(x1)
        x1 = self.conv_1_10(x1)
        x1 = self.conv_1_11(x1)
        x1 = self.conv_1_12(x1)
        x1 = self.conv_1_13(x1)
        x1 = self.conv_1_14(x1)
        x1 = self.conv_1_15(x1)
        x1 = self.conv_1_16(x1)
        x1 = self.conv_1_17(x1)
        x1 = x - x1

        if self.phase == 'train':
            return x1, None
        else:
            return x1


class DNCNN2(nn.Module):
    def __init__(self, in_c, phase):
        super(DNCNN2, self).__init__()
        self.phase = phase
        self.conv_1_1 = nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=64, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        self.conv_1_2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_5 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_6 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_7 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_8 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_9 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_10 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_11 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_12 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_13 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_14 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_15 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_16 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm2d(64), nn.ReLU())

        self.conv_1_17 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=in_c, kernel_size=3, padding=1, bias=False))

    def forward(self, x):
        x1 = self.conv_1_1(x)
        x1 = self.conv_1_2(x1)
        x1 = self.conv_1_3(x1)
        x1 = self.conv_1_4(x1)
        x1 = self.conv_1_5(x1)
        x1 = self.conv_1_6(x1)
        x1 = self.conv_1_7(x1)
        x1 = self.conv_1_8(x1)
        x1 = self.conv_1_9(x1)
        x1 = self.conv_1_10(x1)
        x1 = self.conv_1_11(x1)
        x1 = self.conv_1_12(x1)
        x1 = self.conv_1_13(x1)
        x1 = self.conv_1_14(x1)
        x1 = self.conv_1_15(x1)
        x1 = self.conv_1_16(x1)
        x1 = self.conv_1_17(x1)
        x1 = x + x1

        if self.phase == 'train':
            return x1, None
        else:
            return x1


class DRCNN(nn.Module):
    def __init__(self, in_c, phase):
        super(DRCNN, self).__init__()
        self.phase = phase
        self.conv_1_1 = DNCNN(in_c, phase)

        self.conv_1_2 = DNCNN2(in_c, phase)

    def forward(self, x):
        if self.phase == 'train':
            x1, _ = self.conv_1_1(x)
            x1, _ = self.conv_1_2(x1)
            return x1, None
        else:
            x1 = self.conv_1_1(x)
            x1 = self.conv_1_2(x1)
            return x1


if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    import time
    with torch.no_grad():
        net = DRCNN(3, 'test').cuda()

        f, p = get_model_complexity_info(net, (3, 480, 640), as_strings=True, print_per_layer_stat=False, verbose=False)
        print('FLOPs:', f, 'Parms:', p)

        x = torch.randn(1, 3, 480, 640).cuda()
        s = time.clock()
        y = net(x)
        print(y.shape, 1 / (time.clock() - s))
