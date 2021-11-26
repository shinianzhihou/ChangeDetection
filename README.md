> 本repo基于mmsegmentation进行编写，力求简单易懂好操作。

## 环境配置

1. 基于mmsegmeatation进行编写，所以请先根据官方[repo](./docs/get_started.md#installation)进行环境配置
2. 参照[docker](https://hub.docker.com/r/zenobeijing/pc_pro)配置环境。

> 运行过程中缺少什么package，直接安装即可

## 数据准备
新的数据类为`mmseg/datasets/two_input.py`，主要就是同时输入两个文件，其中有关于数据集的说明。

## 迁移过程
> from mmseg to mmcd

1. 数据部分，为了保证数据增强的多样性以及同时对两张图片进行增强，使用了albumentations替换原本的模块
2. 模型部分
   1. backbone中`two_stream_*`开头的均为修改后的可以用于变化检测的模型
   2. `decoder`部分不变照常使用即可
3. 配置部分，可以参考`configs/cd_stb/*`下面的文件进行查看

## 不足之处

1. 由于数据增强部分用的自己的，所以一些meta信息无法获得（懒得写），所以只能训练过程中并不能进行验证，验证靠`demo/inference_*.ipynb`手动进行。这也是之后的TODO

## 文件夹说明

- configs：配置文件
  - 其中`configs/cd_stb`为昇腾杯的配置文件示例

- demo：一些notebooks
  - 其中`demo/inference_*.ipynb`为推理用的notebook

- tools：一些工具
  - 新加了一些训练脚本，如`tools/train_stb.sh`为训练昇腾杯的启动脚本


## TODO
- [ ] 增加使用例子
- [ ] 增加英文版本
- [ ] 精简文件夹，如config例子
- [ ] merge最新版本的mmseg，加入一些新的transformer模型
- [ ] 数据说明
- [ ] 把收集到的变化检测资源上传上去



**欢迎 Star, issue, PR**
变化检测太苦，马上坚持不下去了呜呜呜呜 (;´༎ຶД༎ຶ`)
（请PR提交到“developing”分支~ ）
