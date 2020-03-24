# Change Detection

一个用 PyTorch 编写的，专门针对变化检测 (Change Detection) 任务的模型框架。

## 写在前面

### 为什么写这个项目？

变化检测（Change Detection，CD）任务与其他任务，如语义分割，目标检测等相比，有其特有的特性（坑），如数据集少（少到可怜那种，尤其是异源，我**），公开的模型也很少，输入常常是成对的（导致一些在 PyTorch 中常用的函数，如Random系列等需要做出一些改变），给初学者带来了很大的困扰（对，没错就是我），所以我将毕设期间写的一些代码，仿照 [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) 整理一下发布出来。

### 特性

- **边训练边测试（可选）**

  由于数据集数量较少，以及 CD 只是一个 “二分类” 问题，所以模型一般较为简单，因此边训练边测试可以更为方便的选到最优解（不增加显存）。

- **“模块式” 搭建模型，方便扩展**

  将整个过程从前期数据处理到后期测试拆分为独立的几个部分，方便之后搭建自己的模型、采用新型的优化器方法或者学习率策略以及增加新的数据集。

- **数据增强**

  将数据增强放在了 “dataloader” 部分，在传统 transform 的基础上实现了对 N 个图片同时操作，保证了 Random 系列图片的一致性

## 开始使用

下表是实现的可以直接用的一些模块（持续更新）

| model                      | dataset  | lr scheduler                                | loss                            | optimizer |
| -------------------------- | -------- | ------------------------------------------- | ------------------------------- | --------- |
| 1. siamese_unet_conc<br /> | 1. Szada | 1. WarmupMultiStepLR<br />2. WarmupCosineLR | 1. BCEWithLogitsLoss+pos_weight | 1. SGD    |

### 0. 数据集

将对应数据集下载并解压到 `data/`目录下

- [Szada]()

  取 7 对大小为 952\*640 的图像的左上角大小为 784\*448 的图片作为测试集，其余部分按照步进为 56 的滑动窗口划分为大小 112\*112 的图片，并以此作为训练集

### 1. 配置文件

按照 `configs/homo/szada_siamese_unet_conc.yaml` 的格式设置自己的配置文件，具体参数说明可以参考 [configs](.CONFIGS.md) 。

### 2. 训练

`-cfg` 指向你的配置文件

```bash
python train_net.py -cfg configs/homo/szada_siamese_unet_conc.yaml
```

### 3. 测试

`-cfg` 指向你的配置文件

```bash
python eval_net.py -cfg configs/homo/szada_siamese_unet_conc.yaml
```

## 结果

| Dataset | Method            | PCC  | Re   | F1   | Kappa | checkpoint                                                   |
| ------- | ----------------- | ---- | ---- | ---- | ----- | ------------------------------------------------------------ |
| Szada   | Siamese_unet_conc | 96.0 | 50.9 | 54.8 | 52.7  | [OneDrive](https://drive.google.com/open?id=17WsyAgMByZB-Rcl5BZiqhoGAlKjTqz1V) |

（单位：%）

### 参考

1. [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
