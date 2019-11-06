#-*-coding:utf-8-*-
# 工具箱
import logging
import os

import pandas as pd


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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='./log.log')
    log = logging.getLogger(__name__)
    log.info('开始执行')
    data = getDataFile('datasets/train/', 'datasets/trainGT/', name='data.csv')
    train, test = split(data, frac=0.6)
    log.info('结束执行')
