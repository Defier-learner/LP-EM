'''
Aechitectures based on Unet which are not currently implemented in archs.py
for Unet3+, this file can be used, but loss function should be carefully checked
Concerning the special architecture of Unet3+, UpSample function has been revised and renamed as DecoderFusion 
and a new class ScaleTransform has been added to handle the feature maps with different sizes in Unet3+.

About loss function: the mixed loss function of BCE and Dice can be used,
but other loss functions should also be concerned
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

class DecoderFusion(nn.Module):
    """Upscaling then double conv"""
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.model=nn.Sequential(
            nn.Conv(in_channels, out_channels, kernel_size=3, padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)

class ScaleTransform(nn.Module):
    #对Unet跨层融合的张亮尺寸进行处理
    def __init__(self, in_channels, out_channels,scale):
        #scale=output_size/input_size, scale>1表示下采样，scale<1表示上采样
        super().__init__()
        layers=[]
        if scale>1:
            #上采样
            layers.append(nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True))
        elif scale<1:
            #下采样
            layers.append(nn.MaxPool2d(kernel_size=int(1/scale), stride=int(1/scale),ceil_mode=True))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.transform=nn.Sequential(*layers)

    def forward(self, x):
        return self.transform(x)

class Unet3plus(nn.Module):
    def __init__(self,num_classes,input_channels=3,deep_supervision=False):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]
        #Encoder与Unet和Unet++相同
        self.inc=DoubleConv(input_channels, nb_filter[0])
        self.down1=DownSample(nb_filter[0], nb_filter[1])
        self.down2=DownSample(nb_filter[1], nb_filter[2])
        self.down3=DownSample(nb_filter[2], nb_filter[3])
        self.down4=DownSample(nb_filter[3], nb_filter[4])

        #Decoder
        self.CatChannels=nb_filter[0]
        self.CatBlocks=5
        self.UpChannels=self.CatChannels*self.CatBlocks

        #hd4
        self.h1_PT_hd4=ScaleTransform(nb_filter[0], self.CatChannels, scale=1/8)
        self.h2_PT_hd4=ScaleTransform(nb_filter[1], self.CatChannels, scale=1/4)
        self.h3_PT_hd4=ScaleTransform(nb_filter[2], self.CatChannels, scale=1/2)
        self.h4_Cat_hd4=ScaleTransform(nb_filter[3], self.CatChannels, scale=1)
        self.h5_UT_hd4=ScaleTransform(nb_filter[4], self.CatChannels, scale=2)
        self.CatBlocks_hd4=DecoderFusion(self.UpChannels, self.UpChannels)

        #hd3
        self.h1_PT_hd3=ScaleTransform(nb_filter[0], self.CatChannels, scale=1/4)
        self.h2_PT_hd3=ScaleTransform(nb_filter[1], self.CatChannels, scale=1/2)
        self.h3_Cat_hd3=ScaleTransform(nb_filter[2], self.CatChannels, scale=1)
        self.d4_UT_hd3=ScaleTransform(self.UpChannels, self.CatChannels, scale=2)
        self.d5_UT_hd3=ScaleTransform(nb_filter[4], self.CatChannels, scale=4)
        self.CatBlocks_hd3=DecoderFusion(self.UpChannels, self.UpChannels)

        #hd2
        self.h1_PT_hd2=ScaleTransform(nb_filter[0], self.CatChannels, scale=1/2)
        self.h2_Cat_hd2=ScaleTransform(nb_filter[1], self.CatChannels, scale=1)
        self.d3_UT_hd2=ScaleTransform(self.UpChannels, self.CatChannels, scale=2)
        self.d4_UT_hd2=ScaleTransform(self.UpChannels, self.CatChannels, scale=4)
        self.d5_UT_hd2=ScaleTransform(nb_filter[4], self.CatChannels, scale=8)
        self.CatBlocks_hd2=DecoderFusion(self.UpChannels, self.UpChannels)

        #hd1
        self.h1_Cat_hd1=ScaleTransform(nb_filter[0], self.CatChannels, scale=1)
        self.d2_UT_hd1=ScaleTransform(self.UpChannels, self.CatChannels, scale=2)
        self.d3_UT_hd1=ScaleTransform(self.UpChannels, self.CatChannels, scale=4)
        self.d4_UT_hd1=ScaleTransform(self.UpChannels, self.CatChannels, scale=8)
        self.d5_UT_hd1=ScaleTransform(nb_filter[4], self.CatChannels, scale=16)
        self.CatBlocks_hd1=DecoderFusion(self.UpChannels, self.UpChannels)

        if deep_supervision:
            self.final1=nn.Conv2d(self.UpChannels, num_classes, kernel_size=1)
            self.final2=nn.Sequential(
                nn.Conv2d(self.UpChannels,num_classes, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )
            self.final3=nn.Sequential(
                nn.Conv2d(self.UpChannels,num_classes, kernel_size=1),
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            )
            self.final4=nn. Sequential(
                nn.Conv2d(self.UpChannels,num_classes, kernel_size=1),
                nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
            )
        else:
            self.final=nn.Conv2d(self.UpChannels, num_classes, kernel_size=1)

    def forward(self, input):
        h1=self.inc(input)
        h2=self.down1(h1)
        h3=self.down2(h2)
        h4=self.down3(h3)
        h5=self.down4(h4)

        #hd4
        h1_PT_hd4=self.h1_PT_hd4(h1)
        h2_PT_hd4=self.h2_PT_hd4(h2)
        h3_PT_hd4=self.h3_PT_hd4(h3)
        h4_Cat_hd4=self.h4_Cat_hd4(h4)
        h5_UT_hd4=self.h5_UT_hd4(h5)
        d4=torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, h5_UT_hd4), dim=1)
        d4=self.CatBlocks_hd4(d4)


        #hd3
        h1_PT_hd3=self.h1_PT_hd3(h1)
        h2_PT_hd3=self.h2_PT_hd3(h2)
        h3_Cat_hd3=self.h3_Cat_hd3(h3)
        d4_UT_hd3=self.d4_UT_hd3(d4)
        d5_UT_hd3=self.d5_UT_hd3(h5)
        d3=torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, d4_UT_hd3, d5_UT_hd3), dim=1)
        d3=self.CatBlocks_hd3(d3)

        #hd2
        h1_PT_hd2=self.h1_PT_hd2(h1)
        h2_Cat_hd2=self.h2_Cat_hd2(h2)
        d3_UT_hd2=self.d3_UT_hd2(d3)
        d4_UT_hd2=self.d4_UT_hd2(d4)
        d5_UT_hd2=self.d5_UT_hd2(h5)
        d2=torch.cat((h1_PT_hd2, h2_Cat_hd2, d3_UT_hd2, d4_UT_hd2, d5_UT_hd2), dim=1)
        d2=self.CatBlocks_hd2(d2)

        #hd1
        h1_Cat_hd1=self.h1_Cat_hd1(h1)
        d2_UT_hd1=self.d2_UT_hd1(d2)
        d3_UT_hd1=self.d3_UT_hd1(d3)
        d4_UT_hd1=self.d4_UT_hd1(d4)
        d5_UT_hd1=self.d5_UT_hd1(h5)
        d1=torch.cat((h1_Cat_hd1, d2_UT_hd1, d3_UT_hd1, d4_UT_hd1, d5_UT_hd1), dim=1)
        d1=self.CatBlocks_hd1(d1)

        if self.deep_supervision:
            output1=self.final1(d1)
            output2=self.final2(d2)
            output3=self.final3(d3)
            output4=self.final4(d4)
            return [output1, output2, output3, output4]#输出前理应考虑加入激活函数
        else:
            return self.final(d1)
