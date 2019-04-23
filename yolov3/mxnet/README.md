模型参考  https://github.com/dmlc/gluon-cv/blob/master/scripts/detection/yolo
# YOLO V3 目标检测

---
## 内容

- [安装](#安装)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型评估](#模型评估)

## 安装

pip install mxnet-cu92

## 数据准备

YOLO v3使用coco数据集进行训练

## 模型训练

python train_yolo3.py --dataset=coco --batch-size=8
在进行性能测试的时候需要测试一个完整的Epoch,计算平均时间

## 模型评估

python eval_yolo.py --dataset=coco --batch-size=8




