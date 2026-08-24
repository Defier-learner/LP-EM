'''
Recurrent Residual Convolutional Neural Network based on U-Net
The model shares the same kernel weights across the recurrent layers, 
which allows the model to learn more complex features and capture long-range dependencies in the input data.

The model should be imported to the register of archs before it can be used in train.py, val.py and test.py
'''

import torch
import torch.nn as nn

class RecurrentBlock(nn.Module):
    def __init__(self, out_channels, t=2):
        super().__init__()
        self.t = t
        self.conv = nn.Sequential(
            #考虑要不要使用偏置
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            else:
                x1 = self.conv(x + x1)
        return x1


class RRCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t=2):
        super().__init__()
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        #两个RecurrentBlock对特征进行深层次的循环提取，可以选择一个，但此处的两个block对应于原始网络的双卷积
    
        self.RCNN = nn.Sequential(
            RecurrentBlock(out_channels, t=t),
            RecurrentBlock(out_channels, t=t)
        )

    def forward(self, x):
        x = self.conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            RRCNNBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = RRCNNBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class R2U_Net(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]
        self.inc = RRCNNBlock(input_channels, nb_filter[0])
        self.down1 = DownSample(nb_filter[0], nb_filter[1])
        self.down2 = DownSample(nb_filter[1], nb_filter[2])
        self.down3 = DownSample(nb_filter[2], nb_filter[3])
        self.down4 = DownSample(nb_filter[3], nb_filter[4])
        self.up1 = UpSample(nb_filter[4] + nb_filter[3], nb_filter[3])
        self.up2 = UpSample(nb_filter[3] + nb_filter[2], nb_filter[2])
        self.up3 = UpSample(nb_filter[2] + nb_filter[1], nb_filter[1])
        self.up4 = UpSample(nb_filter[1] + nb_filter[0], nb_filter[0])
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.inc(input)
        x1_0 = self.down1(x0_0)
        x2_0 = self.down2(x1_0)
        x3_0 = self.down3(x2_0)
        x4_0 = self.down4(x3_0)

        x3_1 = self.up1(x4_0, x3_0)
        x2_2 = self.up2(x3_1, x2_0)
        x1_3 = self.up3(x2_2, x1_0)
        x0_4 = self.up4(x1_3, x0_0)

        output = self.final(x0_4)
        return output