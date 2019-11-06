## 写在前面

- 这个只是小白第一个练手项目，尝试复现的是同实验室师兄的论文`Change Detection Based on Deep Siamese Convolutional Network for Optical Aerial Images`。

- 损失函数方面还没有试到论文的效果，只是用了`pytorch`本身自带的`BCELoss`。

- 如果有什么想法欢迎讨论，但是因为本身是小白，所以可能还会给看的人造成误解，真的很抱歉如果有的话。

## 说明  

一个简单的 Siamese 网络用于**同源**（直方图校正后）图像的变化检测

## 数据集

[原始数据集（未经过处理的）](https://pan.baidu.com/s/1DTn66ygdCuQigFTUB-jMvA)

## 目录  
- Folder

    - `models` : 保存的训练的模型

    - `datasets` : 数据集

    - `tensorboard` : 用于 tensorboard 可视化

- python 脚本

    - `algorithms.py` : 一些简单的算法

    - `configs.py` : 配置文件，包含训练所需的一些超参数

    - `data_loader.py` : 自定义的数据集（继承`torch.utils.data.Dataset`）

    - `models.py` : 定义的网络结构

    - `train.py` : 用来训练网络的脚本

    - `utils.py` : 工具箱，包含一些工具脚本

- csv 文件

    - `data.csv` : 所有的处理后的数据的目录（'img1','img2','GT'）

    - `train.csv` : 训练的数据的位置

    - `test.csv` : 测试的数据的位置

- 其他
    
    - `log.log` : 日志文件

    - `.gitignore` : git 文件，忽略某些文件







