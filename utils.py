#-*-coding:utf-8-*-
# 工具箱
import logging
import os

import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2



def getHeteroDate(imgRoot='datasets/Heterogeneous/',gtRoot='datasets/Heterogeneous/',targetRoot='./',name='heteroData.csv'):
    '''
    此函数用于将异源图像（SAR＆Optical）进行分割得到并存储数据集
    '''
    sar = [] # source image
    opt = [] # source image
    gt = [] # source image
    im1s = [] # processed image
    im2s = [] # processed image
    gts = [] # processed image
    for root,dirs,files in os.walk(imgRoot):
        for file in files:
            if 'sar.png' in file:
                sar.append(root+file)
            elif 'opt.png' in file:
                opt.append(root+file)
            elif 'gt.png' in file and '_' not in file:
                gt.append(root+file)
        break # 不对立面的目录进行遍历
    for img in gts:
        tg = mpimg.imread(img)
        mask = np.concatenate((tg[:,:,0:1],tg[:,:,0:1]))
        



    mask = np.concatenate()

    print(sar,opt,gt,sep='\n--------\n')


def getDataFile(imgRoot=None, gtRoot=None, targetRoot='./', name='data.csv'):
    log.info('开始读取图片并生成csv文件')
    data = pd.DataFrame(columns=['img1', 'img2', 'GT'])
    idx = 0
    # Szada
    for scene in range(1, 8):
        for clip in range(54):
            for view in range(6):
                data.loc[idx] = '{}Szada_Scene{}_img1_Clip{}_View{}.png'.format(
                    imgRoot,scene, clip, view),\
                    '{}Szada_Scene{}_img2_Clip{}_View{}.png'.format(
                    imgRoot,scene, clip, view),\
                    '{}Szada_Scene{}_gt_Clip{}_View{}.png'.format(
                    gtRoot,scene, clip, view)
                log.debug(data.loc[idx])
                idx += 1
    # Tiszadob
    for scene in range(1, 6):
        for clip in range(54):
            for view in range(6):
                data.loc[idx] = '{}Tiszadob_Scene{}_img1_Clip{}_View{}.png'.format(
                    imgRoot,scene, clip, view),\
                    '{}Tiszadob_Scene{}_img2_Clip{}_View{}.png'.format(
                    imgRoot,scene, clip, view),\
                    '{}Tiszadob_Scene{}_gt_Clip{}_View{}.png'.format(
                    gtRoot,scene, clip, view)
                log.debug(data.loc[idx])
                idx += 1
    data.to_csv(targetRoot+name, index=False)
    log.info('已生成csv文件：'+targetRoot+name)

    return data


def split(data=None, frac=0.5, root='./'):
    log.info('开始按照比例'+str(frac)+'进行分割')
    data = data.sample(frac=1.0, replace=False)
    cut = int(data.shape[0]*frac)
    # 注意这里的iloc和loc作用不一样（可能是因为sample后索引被打乱）
    train, test = data.iloc[:cut], data.iloc[cut:]
    train.to_csv(root+'train.csv', index=False)
    test.to_csv(root+'test.csv', index=False)
    log.info('分割结束，在目录'+root+'生成了train.cav和test.csv')
    return train, test

def log(filename='./log.log',level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=filename)
    log = logging.getLogger(__name__)
    return log

if __name__ == "__main__":
    l = log()
    getHeteroDate()   
