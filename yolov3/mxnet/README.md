模型参考  https://github.com/dmlc/gluon-cv/blob/master/scripts/detection/yolo
# YOLO V3 目标检测

---
## 内容

- [安装](#安装)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型评估](#模型评估)

## 安装

在docker环境下进行mxnet安装，在新建docker环境的时候要注意加上--shm-size 16G 的选项，否则会出现内存不足的错误。在docker下使用pip方式安装mxnet根据自己的cuda版本安装相应的mxnet版本，例如cuda9.2版本安装mxnet的命令如下：
    
    pip install mxnet-cu92


## 数据准备

YOLO v3使用coco数据集进行训练(可以使用https://github.com/dmlc/gluon-cv/blob/master/scripts/datasets/mscoco.py进行数据准备)。

## 模型训练

**必要的的依赖** 

    #if gluoncv is not install
    pip install gluoncv
    #if python-tk is not install
    apt-get install python-tk
    #if pycocotools is not install
    pip install pycocotools

**开始训练：** 数据准备完毕后，可以通过如下的方式启动训练：

    python train_yolo3.py --dataset=coco --batch-size=${batch-size} --gpu=${gpu_device}

- 在进行性能测试的时候需要测试一个完整的Epoch,计算平均时间

## 模型评估

python eval_yolo.py --dataset=coco --batch-size=${batch-size} --gpu=${gpu_device}




