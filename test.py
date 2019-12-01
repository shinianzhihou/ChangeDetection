# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, models, transforms
from tensorboardX import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
import logging
import cv2
from math import ceil, sqrt

from configs import Config
from models import *
from data_loader import myData
from utils import log,myshow


# %%
model_list = []
for root,dirs,files in os.walk('./models/Heterogeneous/'):
    for file in files:
        model_list.append(root+file)
for idx,item in enumerate(model_list):
    print("%d:%s"%(idx,item))


# %%
cf = Config()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# %%
net = SiameseUnet(3,2).to(device)
net.load_state_dict(torch.load(model_list[32]))
# net.eval()


# %%
data_transforms = transforms.Compose([
    transforms.ToTensor(),
])

testSet = myData(dataPath='datasets/Heterogeneous/test.csv',
                 transform=data_transforms)

testLoader = DataLoader(dataset=testSet,
                        batch_size=cf.batch_size,
                        shuffle=False,
                        num_workers=cf.num_workers)


# %%
test_res = dict() # 存储最后的结果和原始的gt图片
# 将gt图片和out图片进行拼接和保存
for batch, data in enumerate(testLoader):
    img1, img2, gt, gt_name = data[0].to(device), data[1].to(device), data[2].to(device), data[3]
    output = net(img1,img2)
    out = torch.argmin(output,dim=1, keepdim=True).type_as(gt)
    for idx,item in enumerate(gt_name):
        res = item[item.rfind('/')+1:item.rfind('.png')].split('_')
        res[1:5] = [eval(i) for i in res[1:5]]
        name, R_all, C_all, r, c = res[:5]
        R = ceil(R_all/2)
        C = ceil(C_all/2)
        if name not in test_res.keys():
            test_res[name] = {
                'gt':torch.zeros((R*C,112,112)),
                'out':torch.zeros((R*C,112,112)),
                'row':R,
                'column':C,
                'name':name,
                'TP':-1,
                'TN':-1,
                'FP':-1,
                'FN':-1,
                'F1':-1, # F-Measure
                'Pr':-1, # Precise
                'Re':-1, # Recall
                'PCC':-1, # percentage classification correct
                'Kappa':-1, # Kappa coefficients
            }
        test_res[name]['gt'][(r-1)*C+c-1,:,:] = gt[idx,0,:,:]
        test_res[name]['out'][(r-1)*C+c-1,:,:] = out[idx,0,:,:]


# %%
# 总体的 TP TN FP FN
TP,TN,FP,FN,N = 0,0,0,0,0
# 计算每张图片的指标(加 _ 表示每张图片，不加表示总体数据)
for name,item in test_res.items():
    if 'gt' not in name: # 排除指标项，只留下各个图片的字典
        continue
    # 计算
    TP_ = (item['gt']*item['out']).sum()
    TN_ = ((1-item['gt'])*(1-item['out'])).sum()
    FP_ = ((1-item['gt'])*item['out']).sum()
    FN_ = (item['gt']*(1-item['out'])).sum()
    N_ = TP_ + TN_ + FP_ + FN_
    PCC_ = (TP_+TN_)/N_
    PRE_ = ((TP_+FP_)*(TP_+FN_)+(FN_+TN_)*(FP_+TN_))/(N_*N_)
    Kappa_ =(PCC_-PRE_)/(1-PRE_)
    Pr_ = TP_/(TP_+FP_)
    Re_ = TP_/(TP_+FN_)
    F1_ = 2*Pr_*Re_/(Pr_+Re_)
    # 赋值
    test_res[name]['TP'] = TP_
    test_res[name]['TN'] = TN_
    test_res[name]['FP'] = FP_
    test_res[name]['FN'] = FN_
    test_res[name]['F1'] = F1_
    test_res[name]['Pr'] = Pr_
    test_res[name]['Re'] = Re_
    test_res[name]['PCC'] = PCC_
    test_res[name]['Kappa'] = Kappa_
    # 总体
    TP += TP_
    TN += TN_
    FP += FP_
    FN += FN_
    N += N_

# 计算总体指标
PCC = (TP+TN)/N
PRE = ((TP+FP)*(TP+FN)+(FN+TN)*(FP+TN))/(N*N)
Kappa =(PCC-PRE)/(1-PRE)
Pr = TP/(TP+FP)
Re = TP/(TP+FN)
F1 = 2*Pr*Re/(Pr+Re)
test_res['F1'] = F1
test_res['Pr'] = Pr
test_res['Re'] = Re
test_res['PCC'] = PCC
test_res['kappa'] = Kappa


# %%
test_root = './datasets/Heterogeneous/test/' # 存放测试结果的目录
for name,item in test_res.items():
    if 'gt' not in name:
        continue
    a = item['gt']
    b = item['out']
    a = torch.unsqueeze(a,dim=1)
    b = torch.unsqueeze(b,dim=1)
    a_ = torchvision.utils.make_grid(a,nrow=item['column'],padding=0)
    b_ = torchvision.utils.make_grid(b,nrow=item['column'],padding=0)
    # print(a.shape,a_.shape)
    # print(b.shape,b_.shape)
    torchvision.utils.save_image(a_,test_root+item['name']+'-gt.png',padding=0)
    torchvision.utils.save_image(b_,test_root+item['name']+'-out.png',padding=0)


# %%
# 总体指标
for name,item in test_res.items():
    if 'gt' in name: # 排除不是指标的key
        continue
    print('%s : %f'%(name,item))
# 各个图片的指标
for name,item in test_res.items():
    if 'gt' not in name: # 排除是指标的key
        continue
    print('\n%s :'%(name))
    for idx,value in item.items():
        if idx in['gt','out','row','column','name']:
            continue
        print('%s : %f'%(idx,value))

