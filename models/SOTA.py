import torch
import torch.nn as nn


# 只有最后一层bias=True，其余为False
class ADNET(nn.Module):
    def __init__(self, in_c):
        super(ADNET, self).__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, bias=False, dilation=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_5 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, bias=False, dilation=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_6 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_7 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_8 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_9 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, bias=False, dilation=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_10 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_11 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_12 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, bias=False, dilation=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_13 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_14 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_15 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.conv1_16 = nn.Conv2d(in_channels=64, out_channels=in_c, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=in_c * 2, out_channels=in_c, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
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
        return out2


# 全部bias=True
class BRDNET(nn.Module):
    def __init__(self, in_c):
        super(BRDNET, self).__init__()

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

        return z


# 只有第一层bias=True，其余为False
class DNCNN(nn.Module):
    def __init__(self, in_c):
        super(DNCNN, self).__init__()

        self.conv_1_1 = nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=64, kernel_size=3, padding=1), nn.ReLU(inplace=True))

        self.conv_1_2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_5 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_6 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_7 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_8 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_9 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_10 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_11 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_12 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_13 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_14 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_15 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.conv_1_16 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())

        self.conv_1_17 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=in_c, kernel_size=3, padding=1, bias=False))

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

        return x1


if __name__ == '__main__':
    def print_model_parm_nums(model):
        total = sum([param.nelement() for param in model.parameters()])
        print('  + Number of params: %.2f(e6)' % (total / 1e6))


    x = torch.randn(10, 3, 128, 128).cuda()
    o = BRDNET(3).cuda()
    print_model_parm_nums(o)
    x = o(x)
    print(x.shape)
