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

#非封装写法
class NestedUnet(nn.Module):
    def __init__(self,num_classes, input_channels=3, deep_supervision=False,**kwargs):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]
        self.deep_sepervision=deep_supervision
        self.pool=nn.MaxPool2d(2,2)
        self.up=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        #原始Unet下采样
        self.conv0_0=DoubleConv(input_channels, nb_filter[0])
        self.conv1_0=DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0=DoubleConv(nb_filter[1], nb_filter[2]) 
        self.conv3_0=DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0=DoubleConv(nb_filter[3], nb_filter[4])

        #每个下采样的第一层卷积
        self.conv0_1=DoubleConv(nb_filter[0]+nb_filter[1], nb_filter[0])
        self.conv1_1=DoubleConv(nb_filter[1]+nb_filter[2], nb_filter[1])   
        self.conv2_1=DoubleConv(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv3_1=DoubleConv(nb_filter[3]+nb_filter[4], nb_filter[3])

        #每个下采样的第二层卷积
        self.conv0_2=DoubleConv(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        self.conv1_2=DoubleConv(nb_filter[1]*2+nb_filter[2], nb_filter[1])
        self.conv2_2=DoubleConv(nb_filter[2]*2+nb_filter[3], nb_filter[2])

        #每个下采样的第三层卷积 
        self.conv0_3=DoubleConv(nb_filter[0]*3+nb_filter[1], nb_filter[0])
        self.conv1_3=DoubleConv(nb_filter[1]*3+nb_filter[2], nb_filter[1])

        #每个下采样的第四层卷积
        self.conv0_4=DoubleConv(nb_filter[0]*4+nb_filter[1], nb_filter[0])

        if self.deep_sepervision:   
            self.final1=nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2=nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3=nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4=nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final=nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0=self.conv0_0(input)

        x1_0=self.conv1_0(self.pool(x0_0))
        x0_1=self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1)) 

        x2_0=self.conv2_0(self.pool(x1_0))
        x1_1=self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2=self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0=self.conv3_0(self.pool(x2_0))
        x2_1=self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2=self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3=self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0=self.conv4_0(self.pool(x3_0))
        x3_1=self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2=self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))        
        x1_3=self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))  
        x0_4=self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_sepervision: #考虑是否需要添加激活函数sigmoid  
            output1=self.final1(x0_1)
            output2=self.final2(x0_2)
            output3=self.final3(x0_3)
            output4=self.final4(x0_4)
            return [output1, output2, output3, output4] 
        else:   
            return self.final(x0_4)

        


#对应的，使用封装写法简化

           
