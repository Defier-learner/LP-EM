'''
Aechitectures based on Unet which are not currently implemented in archs.py
for attention unet, this file can be used.
In the attention block, the encoder translates what features are present in the image, 
while the decoder tells what features are needed.
In this way, the attention block can learn to focus on the relevant features in the encoder output 
that are needed for the decoder to make accurate predictions.

the model should be imported to the register of archs before it can be used in train.py, val.py and test.py
'''

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DownSample(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        #g和l谁表示encoder谁表示decoder在具体使用前还需按照不同代码习惯进一步确认
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(self.relu(g1 + x1))
        return x * psi

class UpSample(nn.Module):
    def __init__(self,in_channels,skip_channels,out_channels,F_int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #似乎一种更常见的写法是这里直接进行卷积上采样同时改变通道数
        self.attention = AttentionBlock(F_g=skip_channels, F_l=in_channels, F_int=F_int)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.attention(g=x1, x=x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class AttentionUnet(nn.Module):
    def __init__(self,num_classes,input_channels=3,deep_supervision=False):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]
        self.inc=DoubleConv(input_channels, nb_filter[0])
        self.down1=DownSample(nb_filter[0], nb_filter[1])
        self.down2=DownSample(nb_filter[1], nb_filter[2])
        self.down3=DownSample(nb_filter[2], nb_filter[3])
        self.down4=DownSample(nb_filter[3], nb_filter[4])
        self.up1=UpSample(nb_filter[4], nb_filter[3], nb_filter[3], F_int=nb_filter[3]//2)
        self.up2=UpSample(nb_filter[3], nb_filter[2], nb_filter[2], F_int=nb_filter[2]//2)
        self.up3=UpSample(nb_filter[2], nb_filter[1], nb_filter[1], F_int=nb_filter[1]//2)
        self.up4=UpSample(nb_filter[1], nb_filter[0], nb_filter[0], F_int=nb_filter[0]//2)
        self.outc = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0=self.inc(input)
        x1_0=self.down1(x0_0)
        x2_0=self.down2(x1_0)
        x3_0=self.down3(x2_0)
        x4_0=self.down4(x3_0)

        x3_1=self.up1(x4_0, x3_0)
        x2_2=self.up2(x3_1, x2_0)
        x1_3=self.up3(x2_2, x1_0)
        x0_4=self.up4(x1_3, x0_0)

        output = self.outc(x0_4)
        return output
