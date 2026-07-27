'''
Architecture Rebuild
for Unet, this file can be used
for Unet++ and Unet3+, the code is not complete yet
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

class UpSample(nn.Module):
    """Upscaling then double conv"""
    def __init__(self,in_channels,out_channels):
        super().__init__()
        #ConvTranspose2d会改变通道数，而Upsample不会改变通道数
        #self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1=self.up(x1)
        x=torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self,num_classes,input_channels=3,deep_supervision=False):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]
        self.inc=DoubleConv(input_channels, nb_filter[0])
        self.down1=DownSample(nb_filter[0], nb_filter[1])
        self.down2=DownSample(nb_filter[1], nb_filter[2])
        self.down3=DownSample(nb_filter[2], nb_filter[3])
        self.down4=DownSample(nb_filter[3], nb_filter[4])
        self.up1=UpSample(nb_filter[4]+nb_filter[3], nb_filter[3])
        self.up2=UpSample(nb_filter[3]+nb_filter[2], nb_filter[2])
        self.up3=UpSample(nb_filter[2]+nb_filter[1], nb_filter[1])
        self.up4=UpSample(nb_filter[1]+nb_filter[0], nb_filter[0])
        self.final=nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

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

        output=self.final(x0_4)
        return output

