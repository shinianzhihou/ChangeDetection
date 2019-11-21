import torch

import torchvision
from torch.utils.data import DataLoader,Dataset

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class myData(Dataset):
    def __init__(self,dataPath='./train.csv',transform=None):
        super(myData,self).__init__()
        # data = pd.read_csv(dataPath)
        self.data = pd.read_csv(dataPath)
        self.transform = transform


    def __getitem__(self,index):
        # img1 = self.data.loc[index,'img1']
        # img2 = self.data.loc[index,'img2']
        # gt = self.data.loc[index,'GT']
        transform = self.transform
        img1 = mpimg.imread(self.data.loc[index,'img1'])
        img2 = mpimg.imread(self.data.loc[index,'img2'])
        gt = mpimg.imread(self.data.loc[index,'GT']) # batch_size*1*112*112
        gt_name = self.data.loc[index,'GT']
        # print(gt.shape)
        gt = gt.reshape((112,112,-1))[:,:,0:1]
        # print(gt.shape)

        gt = np.concatenate((gt,1-gt),axis=2)
        # gt = gt.reshape((112,112,1))

        if transform:
            img1 = transform(img1)
            img2 = transform(img2)
            gt = transform(gt)

        return img1,img2,gt,gt_name
    
    def __len__(self):

        return len(self.data)

# if __name__=="__main":
#     mydata = myData()
#     img1,img2,gt = mydata.__getitem__(2)
#     print(gt.shape)