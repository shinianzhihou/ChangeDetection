#-*-coding:utf-8-*-
# 用来存放模型的脚本
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from make_layers import *


class loss(nn.Module):
    def __init__(self):
        super(loss, self).__init__()
        self.m = 5e-4

    def forward(self, D, Y):
        loss = 0.0
        m = self.m  # 可能设置为一个可以学习的参数
        D = D.cpu().detach().numpy()
        Y = Y.cpu().detach().numpy()
        w_u = 0.7
        w_c = 0.3
        loss = np.sum(Y*pow(np.maximum(0, m-D), 2) *
                      w_c + (1-Y)*pow(D, 2)*w_u) / 2

        return torch.tensor(loss, requires_grad=True)


class SiameseUnet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(SiameseUnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # optical image
        self.Oinc = double_conv(n_channels, 64)
        self.Odown1 = Down(64, 128)
        self.Odown2 = Down(128, 256)
        self.Odown3 = Down(256, 512)
        self.Odown4 = Down(512, 512)
        # self.Oup1 = Up(1024, 256, bilinear)
        # self.Oup2 = Up(512, 128, bilinear)
        # self.Oup3 = Up(256, 64, bilinear)
        # self.Oup4 = Up(128, 64, bilinear)
        # self.Ooutc = OutConv(64, n_classes)

        # sar image
        self.Sinc = double_conv(n_channels, 64)
        self.Sdown1 = Down(64, 128)
        self.Sdown2 = Down(128, 256)
        self.Sdown3 = Down(256, 512)
        self.Sdown4 = Down(512, 512)
        self.Sup1 = Up(1024, 256, bilinear)
        self.Sup2 = Up(512, 128, bilinear)
        self.Sup3 = Up(256, 64, bilinear)
        self.Sup4 = Up(128, 64, bilinear)
        self.Soutc = OutConv(64, n_classes)

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
        logits = self.Soutc(z)
        return logits


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


class Siamese_diff_add(nn.Module):
    # 双通道，一个用来学习相同特征，一个用来学习不同特征
    def __init__(self):
        super(Siamese_diff_add, self).__init__()
        self.add_stage = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            #             nn.MaxPool2d(3,stride=2),
            nn.ReLU(),  # inplace=True

            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            #             nn.MaxPool2d(3,stride=2),
            nn.ReLU(),  # inplace=True

            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            #             nn.MaxPool2d(3,stride=2),
            nn.ReLU(),  # inplace=True

        )

        self.diff_stage = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            #             nn.MaxPool2d(3,stride=2),
            nn.ReLU(),  # inplace=True

            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            #             nn.MaxPool2d(3,stride=2),
            nn.ReLU(),  # inplace=True

            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            #             nn.MaxPool2d(3,stride=2),
            nn.ReLU(),  # inplace=True

        )

    def forward(self, im1, im2):
        im_add = (im1 + im2) / 2
        im_diff = (im1 - im2) / 2

        x_add = self.add_stage(im_add)
        x_diff = self.diff_stage(im_diff)

        return


if __name__ == "__main__":
    net = SiameseNet()
    print(net)
    # print(list(net.parameters()))
