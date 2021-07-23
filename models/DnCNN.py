import torch
import torch.nn as nn


# # 只有第一层bias=True，其余为False
# class DNCNN(nn.Module):
#     def __init__(self, in_c, phase):
#         super(DNCNN, self).__init__()
#         self.phase = phase
#         self.conv_1_1 = nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=64, kernel_size=3, padding=1),
#                                       nn.ReLU(inplace=True))

#         self.conv_1_2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
#                                       nn.BatchNorm2d(64), nn.ReLU())
#         self.conv_1_3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
#                                       nn.BatchNorm2d(64), nn.ReLU())
#         self.conv_1_4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
#                                       nn.BatchNorm2d(64), nn.ReLU())
#         self.conv_1_5 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
#                                       nn.BatchNorm2d(64), nn.ReLU())
#         self.conv_1_6 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
#                                       nn.BatchNorm2d(64), nn.ReLU())
#         self.conv_1_7 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
#                                       nn.BatchNorm2d(64), nn.ReLU())
#         self.conv_1_8 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
#                                       nn.BatchNorm2d(64), nn.ReLU())
#         self.conv_1_9 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
#                                       nn.BatchNorm2d(64), nn.ReLU())
#         self.conv_1_10 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
#                                        nn.BatchNorm2d(64), nn.ReLU())
#         self.conv_1_11 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
#                                        nn.BatchNorm2d(64), nn.ReLU())
#         self.conv_1_12 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
#                                        nn.BatchNorm2d(64), nn.ReLU())
#         self.conv_1_13 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
#                                        nn.BatchNorm2d(64), nn.ReLU())
#         self.conv_1_14 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
#                                        nn.BatchNorm2d(64), nn.ReLU())
#         self.conv_1_15 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
#                                        nn.BatchNorm2d(64), nn.ReLU())
#         self.conv_1_16 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
#                                        nn.BatchNorm2d(64), nn.ReLU())

#         self.conv_1_17 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=in_c, kernel_size=3, padding=1, bias=False))

#     def forward(self, x):
#         x1 = self.conv_1_1(x)
#         x1 = self.conv_1_2(x1)
#         x1 = self.conv_1_3(x1)
#         x1 = self.conv_1_4(x1)
#         x1 = self.conv_1_5(x1)
#         x1 = self.conv_1_6(x1)
#         x1 = self.conv_1_7(x1)
#         x1 = self.conv_1_8(x1)
#         x1 = self.conv_1_9(x1)
#         x1 = self.conv_1_10(x1)
#         x1 = self.conv_1_11(x1)
#         x1 = self.conv_1_12(x1)
#         x1 = self.conv_1_13(x1)
#         x1 = self.conv_1_14(x1)
#         x1 = self.conv_1_15(x1)
#         x1 = self.conv_1_16(x1)
#         x1 = self.conv_1_17(x1)
#         x1 = x - x1

#         if self.phase == 'train':
#             return x1, None
#         else:
#             return x1

class DnCNN(nn.Module):
    def __init__(self, channels, phase):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        num_of_layers = 17
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
                                bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)

        if self.phase == 'train':
            return out, None
        else:
            return out

if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    import time
    with torch.no_grad():
        ch = 1
        net = DnCNN(ch, 'test').cuda()

        f, p = get_model_complexity_info(net, (ch, 512, 512), as_strings=True, print_per_layer_stat=False, verbose=False)
        print('FLOPs:', f, 'Parms:', p)

        x = torch.randn(1, ch, 512, 512).cuda()
        s = time.clock()
        y = net(x)
        print(y.shape, 1 / (time.clock() - s))
