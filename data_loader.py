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
        img1 = plt.imread(self.data.loc[index,'img1'])
        img2 = plt.imread(self.data.loc[index,'img2'])
        gt = plt.imread(self.data.loc[index,'GT'])

        if transform:
            img1 = transform(img1)
            img2 = transform(img2)
            gt = transform(gt)

        return (img1,img2),gt
    
    def __len__(self):

        return len(self.data)