# YOLO V3 目标检测

---
## 内容

- [安装](#安装)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型评估](#模型评估)


## 安装

在当前目录下运行样例代码需要PadddlePaddle Fluid的v.1.4或以上的版本。如果你的运行环境中的PaddlePaddle低于此版本，请根据[安装文档](http://paddlepaddle.org/documentation/docs/zh/1.4/beginners_guide/install/index_cn.html)中的说明来更新PaddlePaddle。

## 数据准备

在[MS-COCO数据集](http://cocodataset.org/#download)上进行训练，通过如下方式下载数据集。

    cd dataset/coco
    ./download.sh

数据目录结构如下：

```
dataset/coco/
├── annotations
│   ├── instances_train2014.json
│   ├── instances_train2017.json
│   ├── instances_val2014.json
│   ├── instances_val2017.json
|   ...
├── train2017
│   ├── 000000000009.jpg
│   ├── 000000580008.jpg
|   ...
├── val2017
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
|   ...

```

## 模型训练

**安装[cocoapi](https://github.com/cocodataset/cocoapi)：**

训练前需要首先下载[cocoapi](https://github.com/cocodataset/cocoapi)：

    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonAPI
    # if cython is not installed
    pip install Cython
    # Install into global site-packages
    make install
    # Alternatively, if you do not have permissions or prefer
    # not to install the COCO API into global site-packages
    python2 setup.py install --user

**下载预训练模型：** 本示例提供darknet53预训练模型，该模型转换自作者提供的darknet53在ImageNet上预训练的权重，采用如下命令下载预训练模型：

    sh ./weights/download.sh

通过初始化`pretrain` 加载预训练模型。同时在参数微调时也采用该设置加载已训练模型。
请在训练前确认预训练模型下载与加载正确，否则训练过程中损失可能会出现NAN。

**开始训练：** 数据准备完毕后，可以通过如下的方式启动训练：

    python train.py \
       --model_save_dir=output/ \
       --pretrain=${path_to_pretrain_model}
       --data_dir=${path_to_data}

- 通过设置export CUDA\_VISIBLE\_DEVICES=0,1,2,3,4,5,6,7指定8卡GPU训练。
