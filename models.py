#-*-coding:utf-8-*-
# 用来存放模型的脚本
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


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
        loss = np.sum(Y*pow(np.maximum(0, m-D), 2)*w_c + (1-Y)*pow(D, 2)*w_u) / 2

        return torch.tensor(loss, requires_grad=True)


class SiameseNet(nn.Module):
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


if __name__ == "__main__":
    net = SiameseNet()
    print(net)
    # print(list(net.parameters()))
