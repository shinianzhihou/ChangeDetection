import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnReluDrop(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1,
                 has_bn=True, has_relu=True, has_drop=True, p=0):
        super(ConvBnReluDrop, self).__init__()
        self.has_bn = has_bn
        self.has_relu = has_relu
        self.has_drop = has_drop
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)

        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_c)
        if self.has_relu:
            self.relu = nn.ReLU(inplace = True)
        if self.has_drop:
            self.drop = nn.Dropout2d(p=p)

    def forward(self, x):
        x = self.conv(x)

        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        if self.has_drop:
            x = self.drop(x)
        return x
