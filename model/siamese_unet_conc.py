import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.normal import ConvBnReluDrop


class Siamese_unet_conc(nn.Module):
    def __init__(self, in_c, out_c, p_drop=0.0):
        super(Siamese_unet_conc, self).__init__()

        ####################################################
        # sar and opt
        self.feature_1 = nn.Sequential(
            ConvBnReluDrop(in_c, 16, p=p_drop),
            ConvBnReluDrop(16, 16, p=p_drop),
        )

        self.feature_2 = nn.Sequential(
            ConvBnReluDrop(16, 32, p=p_drop),
            ConvBnReluDrop(32, 32, p=p_drop),
        )

        self.feature_3 = nn.Sequential(
            ConvBnReluDrop(32, 64, p=p_drop),
            ConvBnReluDrop(64, 64, p=p_drop),
            ConvBnReluDrop(64, 64, p=p_drop),
        )

        self.feature_4 = nn.Sequential(
            ConvBnReluDrop(64, 128, p=p_drop),
            ConvBnReluDrop(128, 128, p=p_drop),
            ConvBnReluDrop(128, 128, p=p_drop),
            # ConvBnReluDrop(128,128,p=p_drop),
        )

        ####################################################
        # deconv stage
        self.upconv4 = nn.Sequential(
            ConvBnReluDrop(256, 128, p=p_drop),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
        )

        self.upconv3 = nn.Sequential(
            ConvBnReluDrop(384, 128, p=p_drop),
            ConvBnReluDrop(128, 128, p=p_drop),
            ConvBnReluDrop(128, 64, p=p_drop),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
        )

        self.upconv2 = nn.Sequential(
            ConvBnReluDrop(192, 64, p=p_drop),
            ConvBnReluDrop(64, 64, p=p_drop),
            ConvBnReluDrop(64, 32, p=p_drop),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
        )

        self.upconv1 = nn.Sequential(
            ConvBnReluDrop(96, 32, p=p_drop),
            ConvBnReluDrop(32, 16, p=p_drop),
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
        )

        self.outconv = nn.Sequential(
            ConvBnReluDrop(48, 16, p=p_drop),
            ConvBnReluDrop(16, 2, has_bn=False, has_relu=False, has_drop=False)
        )

    def forward(self, x1, x2):
        ####################################################
        # img1
        x11 = self.feature_1(x1)
        x12 = self.feature_2(F.max_pool2d(x11, kernel_size=2, stride=2))
        x13 = self.feature_3(F.max_pool2d(x12, kernel_size=2, stride=2))
        x14 = self.feature_4(F.max_pool2d(x13, kernel_size=2, stride=2))
        x15 = F.max_pool2d(x14, kernel_size=2, stride=2)
        ####################################################
        # img2
        x21 = self.feature_1(x2)
        x22 = self.feature_2(F.max_pool2d(x21, kernel_size=2, stride=2))
        x23 = self.feature_3(F.max_pool2d(x22, kernel_size=2, stride=2))
        x24 = self.feature_4(F.max_pool2d(x23, kernel_size=2, stride=2))
        x25 = F.max_pool2d(x24, kernel_size=2, stride=2)
        ####################################################
        # fusion(x1,x2,x3) and up
        x_f4 = torch.cat([x15, x25], dim=1)
        x34 = self.upconv4(x_f4)
        x_f3 = torch.cat([x34, x14, x24], dim=1)
        x33 = self.upconv3(x_f3)
        x_f2 = torch.cat([x33, x13, x23], dim=1)
        x32 = self.upconv2(x_f2)
        x_f1 = torch.cat([x32, x12, x22], dim=1)
        x31 = self.upconv1(x_f1)
        x_fout = torch.cat([x31, x11, x21], dim=1)
        output = self.outconv(x_fout)


        return output
