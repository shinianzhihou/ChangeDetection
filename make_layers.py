import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

## unet
class double_conv(nn.Module):
    def __init__(self,in_C,out_C):
        super(double_conv,self).__init__()
        self.conv_conv = nn.Sequential(

            nn.Conv2d(in_C,out_C,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_C),
            nn.ReLU(inplace=False),

            nn.Conv2d(out_C,out_C,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_C),
            nn.ReLU(inplace=False),

        )

    def forward(self,x):
        return self.conv_conv(x)


class Down(nn.Module):
    # 下采样后卷积两次（不先卷积的原因是需要卷积的结果进行上采样）
    def __init__(self,in_C,out_C):
        super(Down,self).__init__()
        self.maxpool_double_conv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_C,out_C)
        )

    def forward(self,x):
        return self.maxpool_double_conv(x)

class Up(nn.Module):
    # 上采样后卷积两次

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = double_conv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

