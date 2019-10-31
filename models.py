#-*-coding:utf-8-*-
# 用来存放模型的脚本
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        

    def forward(self,x1,x2):
        x1 = self.featureExtract(x1)
        x2 = self.featureExtract(x2)        
        x = F.pairwise_distance(x1,x2,p=2) # 2范数
        
        return x


if __name__=="__main__":
    net = SiameseNet()
    print(net)
    # print(list(net.parameters()))