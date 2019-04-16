# lstm lm

以下是本例的简要目录结构及说明：

```text
.
├── README.md            # 文档
├── train.py             # 训练脚本
├── reader.py            # 数据读取
└── ptb_lm_model.py             # 模型定义文件
```


## 简介

循环神经网络语言模型的介绍可以参阅论文[Recurrent Neural Network Regularization](https://arxiv.org/abs/1409.2329)，本文主要是说明基于lstm的语言的模型的实现，数据是采用ptb dataset，下载地址为
http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

## 数据下载
用户可以自行下载数据，并解压， 也可以利用目录中的脚本

参考tf文档https://www.tensorflow.org/tutorials/sequences/recurrent

## 训练
修改run.sh中的参数为
`CUDA_VISIBLE_DEVICES='0' python train.py --model_type large --inference_only False`
运行命令
bash run.sh
开始单卡训练模型。

修改run.sh中的参数为
`CUDA_VISIBLE_DEVICES='' python train.py --model_type large --inference_only False`
运行命令
bash run.sh
开始cpu训练模型

##预测
修改run.sh中的参数为
`CUDA_VISIBLE_DEVICES='0' python train.py --model_type large --inference_only True`
运行命令
bash run.sh
开始单卡预测模型

修改run.sh中的参数为
`CUDA_VISIBLE_DEVICES='' python train.py --model_type large --inference_only True`
运行命令
bash run.sh
开始cpu预测模型。

实现采用双层的lstm，具体的参数和网络配置 可以参考 train.py， ptb_lm_model.py 文件中的设置.


## 与tf结果对比

tf采用的版本是1.12
large config



|         |单卡训练速度|单卡训练显存占用（最大batchsize)|单卡GPU预测速度（batch_size=1)|单卡GPU预测显存占用(batch_size=1)|
| --------   | -----:   | :----: |:----: |:----: |
|fluid dev|146/epoch|500|236s|6121MB|
|tf 1.12  |83/epoch|1642|46s|1095MB|
