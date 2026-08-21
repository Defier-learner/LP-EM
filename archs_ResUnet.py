'''
ResUnet Architecture

ResUnet: residual net is a net which learns the difference between the input and the output. 
Unet: x->H(x)
ResUnet: x->F(x)+x where F(x)=H(x)-x, thus F(x) represents the residual mapping to be learned.
The maintaining of the identity mapping in ResUnet allows the network to learn more complex features.
For example, the ibput image is composed of weak features and strong noises,
the Unet will learn the strong noises and weak features together, while the ResUnet will learn the weak features and ignore the strong noises.
which avoids the loss of weak features during the convolution process.
Dynamic learning rate should be considered

Special tips: pre-activation architecture is used in ResUnet, which means that the order of the convolutional layer and the activation function is reversed.
This enables the maximum preservation of the original information in the input image, especially in the decoder process.
Also, classical implementation of ResUnet should be considered.

Similarly,the model should be imported to the register of archs before it can be used in train.py, val.py and test.py
'''

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        if in_channels != out_channels:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.short_cut = nn.Identity()

    def forward(self, x):
        return self.conv_block(x) + self.short_cut(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ResidualBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ResUnet(nn.Module):
    def __init__(self, num_classes, input_channels=3,deep_supervision=False):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]
        self.inc=nn.Sequential(
            nn.Conv2d(input_channels, nb_filter[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(nb_filter[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_filter[0], nb_filter[0], kernel_size=3, padding=1)
        )
        self.inc_short_cut=nn.Sequential(
            nn.Conv2d(input_channels, nb_filter[0], kernel_size=1,bias=False),
            nn.BatchNorm2d(nb_filter[0])
        )
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
        x0_0 = self.inc(input)+self.inc_short_cut(input)
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
