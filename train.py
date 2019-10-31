import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import matplotlib.pyplot as plt

import os

from configs import Config
from models import SiameseNet
from data_loader import myData


# configuration
cf = Config()
batch_size = cf.batch_size
num_epochs = cf.num_epochs
num_workers = cf.num_workers
leaning_rate = cf.learning_rate
momentum = cf.momentum
weight_decay = cf.weight_decay

# dataset
data_transforms = transforms.Compose([
    transforms.ToTensor(),
])

trainSet = myData(dataPath='./train.csv',
                  transform=data_transforms)
testSet = myData(dataPath='./test.csv',
                 transform=data_transforms)

trainLoader = DataLoader(dataset=trainSet,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=num_workers)
testLoader = DataLoader(dataset=testSet,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers)

# net
net = SiameseNet()
# gpu
if cf.use_gpu:
    net.to('cuda')

net.train()

# optimizer
# optimizer = optim.Adam(net.parameters(),lr=leaning_rate)
optimizer = optim.SGD(net.parameters(), lr=leaning_rate,
                      momentum=momentum, weight_decay=weight_decay)
optimizer.zero_grad()

for i in range(num_epochs):
