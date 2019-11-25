#-*-coding:utf-8-*-
# 用来存放模型的脚本
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from make_layers import *


class SiameseNet(nn.Module):
    # 卷积过程中feature map大小不变的卷积网络
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.featureExtract = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.Conv2d(64, 32, 5, padding=2),
            nn.Conv2d(32, 16, 1, padding=0),
        )
        self.distance = nn.PairwiseDistance(p=2)

    def forward(self, x1, x2):
        x1 = self.featureExtract(x1)
        x2 = self.featureExtract(x2)
        x = self.distance(x1, x2)

        return x



class loss(nn.Module):
    def __init__(self):
        super(loss, self).__init__()
        self.m = 1e-5  # 一个极小的值




    def forward(self, Y, D):
        D = D.cpu().detach().numpy()  # gt
        Y = Y.cpu().detach().numpy()  # output

        m = self.m
        loss = 0.0
        # print(D.shape,Y.shape)
        weights = [0.7, 0.3]  # 对不同mask进行加权
        for mask_idx, weight in enumerate(weights):
            # mask 表示 gt 中的不同 mask 0,1，...
            mask = D[:, mask_idx, :, :]
            out = Y[:, mask_idx, :, :]
            loss += (- weight * np.mean(mask*np.log(out+m) +
                                        (1-mask)*np.log(1-out+m)))

        return torch.tensor(loss, requires_grad=True)


class SiameseUnet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(SiameseUnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # optical image
        self.Oinc = double_conv(n_channels, 16)
        self.Odown1 = Down(16, 32)
        self.Odown2 = Down(32, 64)
        self.Odown3 = Down(64, 128)
        self.Odown4 = Down(128, 128)
        # self.Oup1 = Up(1024, 256, bilinear)
        # self.Oup2 = Up(512, 128, bilinear)
        # self.Oup3 = Up(256, 64, bilinear)
        # self.Oup4 = Up(128, 64, bilinear)
        # self.Ooutc = OutConv(64, n_classes)

        # sar image
        self.Sinc = double_conv(n_channels, 16)
        self.Sdown1 = Down(16, 32)
        self.Sdown2 = Down(32, 64)
        self.Sdown3 = Down(64, 128)
        self.Sdown4 = Down(128, 128)
        self.Sup1 = Up(256, 64, bilinear)
        self.Sup2 = Up(128, 32, bilinear)
        self.Sup3 = Up(64, 16, bilinear)
        self.Sup4 = Up(32, 16, bilinear)
        self.Soutc = OutConv(16, n_classes)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x, y):
        # Optical
        x1 = self.Oinc(x)
        x2 = self.Odown1(x1)
        x3 = self.Odown2(x2)
        x4 = self.Odown3(x3)
        # x5 = self.Odown4(x4)
        # SAR
        y1 = self.Sinc(y)
        y2 = self.Sdown1(y1)
        y3 = self.Sdown2(y2)
        y4 = self.Sdown3(y3)
        y5 = self.Sdown4(y4)

        # Optical 更容易解决 “where”
        z = self.Sup1(y5, x4)
        z = self.Sup2(z, x3)
        z = self.Sup3(z, x2)
        z = self.Sup4(z, x1)
        z = self.Soutc(z)
        z = self.soft(z)
        # print(z.type())

        return z

class myloss(nn.Module):
    def __init__(self):
        super(myloss,self).__init__()
    
    def forward(self,input,target):
        return torch.mean(target*torch.log(input)+(1-target)*torch.log(1-target))


if __name__ == "__main__":
    net = SiameseNet()
    print(net)
    # print(list(net.parameters()))
