import torch
import torch.nn as nn


# 全部bias=True
class BRDNET(nn.Module):
    def __init__(self, in_c, phase):
        super(BRDNET, self).__init__()
        self.phase = phase
        self.conv_1_1 = nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.conv_1_2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv_1_3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv_1_4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv_1_5 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv_1_6 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv_1_7 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv_1_8 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.conv_1_9 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv_1_10 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv_1_11 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv_1_12 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv_1_13 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv_1_14 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv_1_15 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv_1_16 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.conv_1_17 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=in_c, kernel_size=3, padding=1))
        ###############################
        self.conv_2_1 = nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.conv_2_2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.conv_2_3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.conv_2_4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.conv_2_5 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.conv_2_6 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.conv_2_7 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.conv_2_8 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True))

        self.conv_2_9 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.conv_2_10 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.conv_2_11 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.conv_2_12 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.conv_2_13 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.conv_2_14 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.conv_2_15 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True))

        self.conv_2_16 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.conv_2_17 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=in_c, kernel_size=3, padding=1))
        ###############################
        self.conv_6_to_3 = nn.Sequential(nn.Conv2d(in_channels=in_c * 2, out_channels=in_c, kernel_size=3, padding=1))

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

        y1 = self.conv_2_1(x)
        y1 = self.conv_2_2(y1)
        y1 = self.conv_2_3(y1)
        y1 = self.conv_2_4(y1)
        y1 = self.conv_2_5(y1)
        y1 = self.conv_2_6(y1)
        y1 = self.conv_2_7(y1)
        y1 = self.conv_2_8(y1)
        y1 = self.conv_2_9(y1)
        y1 = self.conv_2_10(y1)
        y1 = self.conv_2_11(y1)
        y1 = self.conv_2_12(y1)
        y1 = self.conv_2_13(y1)
        y1 = self.conv_2_14(y1)
        y1 = self.conv_2_15(y1)
        y1 = self.conv_2_16(y1)
        y1 = self.conv_2_17(y1)
        y1 = x - y1

        z = torch.cat([x1, y1], 1)
        z = self.conv_6_to_3(z)
        z = x - z

        if self.phase == 'train':
            return z, None
        else:
            return z




if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    import time
    with torch.no_grad():
        net = BRDNET(3, 'test').cuda()

        f, p = get_model_complexity_info(net, (3, 480, 640), as_strings=True, print_per_layer_stat=False, verbose=False)
        print('FLOPs:', f, 'Parms:', p)

        x = torch.randn(1, 3, 480, 640).cuda()
        s = time.clock()
        y = net(x)
        print(y.shape, 1 / (time.clock() - s))