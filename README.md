# PaddlePaddle Benchmark

我们对PaddlePaddle的最新版本[v1.5.0](https://github.com/PaddlePaddle/Paddle/tree/v1.5.0)，在训练性能和显存占用方面进行了基准测试。

## 目录
* [测试环境](#测试环境)
* [智能视觉(PaddleCV)](#PaddleCV)
  * [SE-ResNeXt50](#SE-ResNeXt50)
  * [Mask-RCNN](#Mask-RCNN)
  * [YOLOv3](#YOLOv3)
  * [DeepLab V3+](#DeepLab V3+)
  * [CycleGAN](#CycleGAN)
* [智能文本处理(PaddleNLP)](#PaddleNLP)
  * [PaddingRNN](#PaddingRNN)
  * [BERT](#BERT)
  * [Transformer](#Transformer)
* [强化学习(PARL)](#PARL)
  * [DDPG](#DDPG)

## 测试环境
- 测试对象
  - 本次测试[PaddlePaddle v1.5.0](https://github.com/PaddlePaddle/Paddle/tree/v1.5.0)，具体commit是：`401c03fc20478f5cc067440422fc3a7b306d0e32`
  - 基准测试程序[benchmark](https://github.com/PaddlePaddle/benchmark)，具体commit是：`3c34ed6b166f6b77e759b4c54e8854652ad3d776`


- Docker镜像
  - Paddle编译镜像
    - CUDA 9.0，`paddlepaddle/paddle_manylinux_devel:cuda9.0_cudnn7`
    - CUDA 10.0，`paddlepaddle/paddle_manylinux_devel:cuda10.0_cudnn7`
  - Paddle测试镜像
    - CUDA 9.0，`paddlepaddle/paddle:latest-gpu-cuda9.0-cudnn7`
    - CUDA 10.0，`paddlepaddle/paddle:latest-gpu-cuda10.0-cudnn7`
  - TensorFlow测试镜像
    - CUDA 9.0，`tensorflow/tensorflow:1.12.0-gpu`
    - CUDA 10.0，`tensorflow/tensorflow:1.14.0-gpu`
  - PyTorch
    - CUDA 9.0，
    - CUDA 10.0，

- GPU服务器参数
  - GPU型号：Nvidia Tesla V100-SXM2，显存16 GB
  - CPU型号：Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz，38核
  - Driver Version: 418.39
  - CUDA Version：9.0.176，10.0.130
  - NCCL Version：2.4.2
  - cuDNN Version：7.4.2.24，7.5.0.56

- CPU服务器参数
  - CPU型号：Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz，24核
  - 指令集：AVX2

## PaddleCV

| 方向 | 模型 | Paddle | TensorFlow | PyTorch | MXNet | 数据集 | batch_size(单卡) |
|---|---|---|---|---|---|---|---|
| 图像分类 | SE-ResNeXt50 | [PaddleCV/image_classification](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification) | - | [SENet-PyTorch](https://github.com/miraclewkf/SENet-PyTorch) | - | ILSVRC2012 | 32 |
| 目标检测 | Mask-RCNN | [PaddleCV/rcnn](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/rcnn) | - | [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) | - | COCO17 | 1 |
| 目标检测 | YOLOv3 | [Paddle/yolov3](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/yolov3) | - | - | [gluon-cv](https://github.com/dmlc/gluon-cv/tree/master/scripts/detection/yolo) | COCO17 | 8 |
| 图像分割 | DeepLab V3+ | [PaddleCV/deeplabv3+](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/deeplabv3%2B) | [tensorflow/models](https://github.com/tensorflow/models/tree/master/research/deeplab) | - | - | cityscape | 2 |
| 图像生成 | CycleGAN | [PaddleCV/PaddleGAN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleGAN/cycle_gan) | [CycleGAN](https://github.com/hardikbansal/CycleGAN) | - | - | horse2zebra | 1 |

### SE-ResNeXt50
SE-ResNeXt50模型单卡训练速度与PyTorch**持平**，八卡训练速度和显存占用都**优于**PyTorch。

- 准备工作
- 训练速度

单位：images/s

1.&nbsp;**CUDA 9.0**测试结果 

| ` ` | Paddle 1.5.0 | PyTorch 1.1.0 |
|---|---|---|
| 1 GPU | 168.334 | 163.130 |
| 8 GPUs (单进程) | 843.348 | 595.274 |
| 8 GPUs (多进程) | - | - |

2.&nbsp;**CUDA 10.0**测试结果 

| ` ` | Paddle 1.5.0 | PyTorch 1.1.0 |
|---|---|---|
| 1 GPU | 168.478 | 163.294 |
| 8 GPUs (单进程) | 836.357 | 573.732 |
| 8 GPUs (多进程) | - | - |

- 显存占用

1.&nbsp;**CUDA 9.0**测试结果

| ` ` | Paddle 1.5.0 | PyTorch 1.1.0 |
|---|---|---|
| 单卡显存占用 | 5515 MiB | 5677 MiB |
| 单卡最大batch_size | 112 | 112 |

2.&nbsp;**CUDA 10.0**测试结果 

| ` ` | Paddle 1.5.0 | PyTorch 1.1.0 |
|---|---|---|
| 单卡显存占用 | 5535 MiB | 5695 MiB |
| 单卡最大batch_size | 112 | 112 |

### Mask-RCNN
Mask-RCNN模型训练速度和显存占用都**优于**PyTorch。

- 准备工作
- 训练速度

单位：images/s

1.&nbsp;**CUDA 9.0**测试结果

| ` ` | Paddle 1.5.0 | PyTorch 1.1.0 |
|---|---|---|
| 1 GPU | 3.811 | 3.240 |
| 8 GPUs (单进程) | 18.707 | - |
| 8 GPUs (多进程) | 23.014 | 21.864 |

2.&nbsp;**CUDA 10.0**测试结果  

| ` ` | Paddle 1.5.0 | PyTorch 1.1.0 |
|---|---|---|
| 1 GPU | 3.780 | - |
| 8 GPUs (单进程) | 18.505 | - |
| 8 GPUs (多进程) | 23.199 | - |

- 显存占用

1.&nbsp;**CUDA 9.0**测试结果

| ` ` | Paddle 1.5.0 | PyTorch 1.1.0 |
|---|---|---|
| 单卡显存占用 | 3871 MiB | 4548 MiB |
| 单卡最大batch_size | 5 | 5 |

2.&nbsp;**CUDA 10.0**测试结果

| ` ` | Paddle 1.5.0 | PyTorch 1.1.0 |
|---|---|---|
| 单卡显存占用 | 3907 MiB | - |
| 单卡最大batch_size | 5 | - |

### YOLOv3
YOLOv3模型训练速度和显存占用都**优于**MXNet。

- 准备工作
- 训练速度

单位：images/s
  
1.&nbsp;**CUDA 9.0**测试结果

| ` ` | Paddle 1.5.0 | MXNet |
|---|---|---|
| 1 GPU | 29.901 | 18.578 |
| 8 GPUs (单进程) | 58.175 | 35.574 |
| 8 GPUs (多进程) | 99.530 | - |

2.&nbsp;**CUDA 10.0**测试结果

| ` ` | Paddle 1.5.0 | MXNet |
|---|---|---|
| 1 GPU | 30.591 | 17.001 |
| 8 GPUs (单进程) | 57.997 | 33.755 |
| 8 GPUs (多进程) | 104.553 | - |

- 显存占用

1.&nbsp;**CUDA 9.0**测试结果

| ` ` | Paddle 1.5.0 | PyTorch 1.1.0 |
|---|---|---|
| 单卡显存占用 | 10583 MiB | 14304 MiB |
| 单卡最大batch_size | 14 | 14 |

2.&nbsp;**CUDA 10.0**测试结果

| ` ` | Paddle 1.5.0 | PyTorch 1.1.0 |
|---|---|---|
| 单卡显存占用 | 10599 MiB | 9842 MiB |
| 单卡最大batch_size | 14 | 14 |

### DeepLab V3+
Deep Lab V3+模型训练速度和显存占用都**优于**TensorFlow。

- 准备工作
- 训练速度

单位：images/s

1.&nbsp;**CUDA 9.0**测试结果

| ` ` | Paddle 1.5.0 | TensorFlow 1.12.0 |
|---|---|---|
| 1 GPU | 13.695 | 6.4 |
| 8 GPUs (单进程) | 59.721 | 16.508 |

2.&nbsp;**CUDA 10.0**测试结果  

| ` ` | Paddle 1.5.0 | TensorFlow 1.14.0 |
|---|---|---|
| 1 GPU | 14.261 | 6.309 |
| 8 GPUs (单进程) | 58.024 | 16.427 |

- 显存占用

1.&nbsp;**CUDA 9.0**测试结果

| ` ` | Paddle 1.5.0 | TensorFlow 1.12.0 |
|---|---|---|
| 单卡显存占用 | 5163 MiB | 8934 MiB |
| 单卡最大batch_size | 9 | 7 |

2.&nbsp;**CUDA 10.0**测试结果

| ` ` | Paddle 1.5.0 | TensorFlow 1.14.0 |
|---|---|---|
| 单卡显存占用 | 5167 MiB | 8927 MiB |
| 单卡最大batch_size | 9 | 7 |

### CycleGAN
Cycle-GAN模型不支持多卡训练，其单卡训练速度和显存占用都**优于**TensorFlow。

- 准备工作
- 训练速度

单位：images/s

1.&nbsp;**CUDA 9.0**测试结果

| ` ` | Paddle 1.5.0 | TensorFlow 1.12.0 |
|---|---|---|
| 1 GPU | 7.513 | 6.452 |

2.&nbsp;**CUDA 10.0**测试结果  

| ` ` | Paddle 1.5.0 | TensorFlow 1.14.0 |
|---|---|---|
| 1 GPU | 7.591 | 6.823 |

- 显存占用

1.&nbsp;**CUDA 9.0**测试结果

| ` ` | Paddle 1.5.0 | TensorFlow 1.12.0 |
|---|---|---|
| 单卡显存占用 | 2479 MiB | 5094 MiB |

2.&nbsp;**CUDA 10.0**测试结果

| ` ` | Paddle 1.5.0 | TensorFlow 1.14.0 |
|---|---|---|
| 单卡显存占用 | 2499 MiB | 5089 MiB |

## PaddleNLP

| 方向 | 模型 | Paddle | TensorFlow | PyTorch | 数据集 | batch_size(单卡) |
|---|---|---|---|---|---|---|
| 语言模型 | PaddingRNN | [PaddleNLP/language_model](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/language_model) | [benchmark/PaddingRNN/lstm_tf](https://github.com/PaddlePaddle/benchmark/tree/master/PaddingRNN/lstm_tf) | - | PTB文本数据集 | 20 |
| 语义表示 | BERT | [LARK](https://github.com/PaddlePaddle/LARK) | [google-research/bert](https://github.com/google-research/bert) | - | XNLI | 32 |
| 机器翻译 | Transformer | [PaddleNLP/neural_machine_translation/transformer](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/neural_machine_translation/transformer) | [tensor2tensor](https://github.com/tensorflow/tensor2tensor) | - | En-de | 4096 |

### PaddingRNN
TensorFlow的PaddingRNN开源模型多卡训练失败，故只测试单卡训练的情况。
PaddleRNN模型在static模式下，单卡训练速度和显存占用都**差于**TensorFlow。

- 准备工作
- 训练速度

单位：steps/s

1.&nbsp;static模式，small模型，**CUDA 9.0**测试结果

| ` ` | Paddle 1.5.0 | TensorFlow 1.12.0 |
|---|---|---|
| 1 GPU | 61.208 | 73.991 |

2.&nbsp;static模式，small模型，**CUDA 10.0**测试结果  

| ` ` | Paddle 1.5.0 | TensorFlow 1.14.0 |
|---|---|---|
| 1 GPU | 63.400 | 72.406 |

3.&nbsp;static模式，large模型，**CUDA 9.0**测试结果

| ` ` | Paddle 1.5.0 | TensorFlow 1.12.0 |
|---|---|---|
| 1 GPU | 17.479 | 18.529 |

4.&nbsp;static模式，large模型，**CUDA 10.0**测试结果  

| ` ` | Paddle 1.5.0 | TensorFlow 1.14.0 |
|---|---|---|
| 1 GPU | 17.107 | 17.914 |

- 显存占用

1.&nbsp;static模式，small模型，**CUDA 9.0**测试结果

| ` ` | Paddle 1.5.0 | TensorFlow 1.12.0 |
|---|---|---|
| 单卡显存占用 | 660 MiB | 660 MiB |

2.&nbsp;static模式，small模型，**CUDA 10.0**测试结果

| ` ` | Paddle 1.5.0 | TensorFlow 1.14.0 |
|---|---|---|
| 单卡显存占用 | 657 MiB | 647 MiB |

3.&nbsp;static模式，large模型，**CUDA 9.0**测试结果

| ` ` | Paddle 1.5.0 | TensorFlow 1.12.0 |
|---|---|---|
| 单卡显存占用 | 6089 MiB | 5858 MiB |

4.&nbsp;static模式，large模型，**CUDA 10.0**测试结果

| ` ` | Paddle 1.5.0 | TensorFlow 1.14.0 |
|---|---|---|
| 单卡显存占用 | 6083 MiB | 8711 MiB |

### BERT
TensorFlow的BERT开源模型暂无多卡实现。
BERT模型单卡训练速度和显存占用都优于TensorFlow。

- 准备工作
- 训练速度

单位：steps/s

1.&nbsp;**CUDA 9.0**测试结果

| ` ` | Paddle 1.5.0 | TensorFlow 1.12.0 |
|---|---|---|
| 1 GPU | 4.044 | 3.420 |
| 8 GPUs (单进程) | 1.803 | - |
| 8 GPUs (多进程) | 3.114 | - |

2.&nbsp;**CUDA 10.0**测试结果  

| ` ` | Paddle 1.5.0 | TensorFlow 1.14.0 |
|---|---|---|
| 1 GPU | 4.003 | - |
| 8 GPUs (单进程) | 1.817 | - |
| 8 GPUs (多进程) | 3.089 | - |

- 显存占用

1.&nbsp;**CUDA 9.0**测试结果

| ` ` | Paddle 1.5.0 | TensorFl 1.12.0 |
|---|---|---|
| 单卡显存占用 | 6551 MiB | 15430 MiB |
| 单卡最大batch_size | 9984 | 9216 |

2.&nbsp;**CUDA 10.0**测试结果

| ` ` | Paddle 1.5.0 | TensorFlow 1.14.0 |
|---|---|---|
| 单卡显存占用 | 6545 MiB | - |
| 单卡最大batch_size | 9984 | - |

### Transformer
Transformer模型单卡训练速度与TensorFlow**持平**；多卡训练速度和显存占用**优于**TensorFlow。

- 准备工作
- 训练速度

单位：steps/s

1.&nbsp;**CUDA 9.0**测试结果

| ` ` | Paddle 1.5.0 | TensorFlow 1.12.0 |
|---|---|---|
| 1 GPU | 4.865 | 4.750 |
| 8 GPUs (单进程) | 4.227 | 2.302 |
| 8 GPUs (多进程) | 4.445 | - |

2.&nbsp;**CUDA 10.0**测试结果  

| ` ` | Paddle 1.5.0 | TensorFlow 1.14.0 |
|---|---|---|
| 1 GPU | 4.883 | 4.721 |
| 8 GPUs (单进程) | 4.355 | 2.520 |
| 8 GPUs (多进程) | 4.460 | - |

- 显存占用

1.&nbsp;**CUDA 9.0**测试结果

| ` ` | Paddle 1.5.0 | TensorFl 1.12.0 |
|---|---|---|
| 单卡显存占用 | 7137 MiB | 8948 MiB |
| 单卡最大batch_size | 12000 | 11144 |

2.&nbsp;**CUDA 10.0**测试结果

| ` ` | Paddle 1.5.0 | TensorFlow 1.14.0 |
|---|---|---|
| 单卡显存占用 | 7147 MiB | 8711 |
| 单卡最大batch_size | 12000 | 11144 |

## PARL

| 方向 | 模型 | Paddle | TensorFlow | PyTorch | 数据集 | batch_size(单卡) |
|---|---|---|---|---|---|---|
| 强化学习 | DDPG Deep Explore |	[benchmark/DDPG_Deep_Explore/Fluid_version](https://github.com/PaddlePaddle/benchmark/tree/master/DDPG_Deep_Explore/Fluid_version) | [benchmark/DDPG_Deep_Explore/TF_version](https://github.com/PaddlePaddle/benchmark/tree/master/DDPG_Deep_Explore/TF_version) | - | 测试数据 | 100 |

### DDPG
DDPG模型不支持多卡训练，其训练速度与竞品持平，显存占用**优于**TensorFlow。

- 准备工作
- 训练速度

单位：epoch/s

1.&nbsp;**CUDA 9.0**测试结果

| ` ` | Paddle 1.5.0 | TensorFlow 1.12.0 |
|---|---|---|
| 1 GPU | 1.615 | 1.606 |

2.&nbsp;**CUDA 10.0**测试结果

| ` ` | Paddle 1.5.0 | TensorFlow 1.14.0 |
|---|---|---|
| 1 GPU | 1.578 | - |

- 显存占用

1.&nbsp;**CUDA 9.0**测试结果

| ` ` | Paddle 1.5.0 | TensorFlow 1.12.0 |
|---|---|---|
| 单卡显存占用 | 563 MiB | 630 MiB |

2.&nbsp;**CUDA 10.0**测试结果

| ` ` | Paddle 1.5.0 | TensorFlow 1.14.0 |
|---|---|---|
| 单卡显存占用 | 3907 MiB | - |