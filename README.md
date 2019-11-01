## 说明  

一个简单的 Siamese 网络用于同源变化检测

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







