# PaddlePaddle Benchmark

我们对PaddlePaddle的最新版本[v1.5.0](https://github.com/PaddlePaddle/Paddle/tree/v1.5.0)，在训练性能和显存占用方面进行了基准测试。

## 目录
* [测试环境](#测试环境)
* [智能视觉（PaddleCV）](#PaddleCV)
  * [SE-ResNeXt50](#SE-ResNeXt50)
  * [Mask-RCNN](#Mask-RCNN)
  * [YOLOv3](#YOLOv3)
  * [DeepLab V3+](#deepLab-v3)
  * [Cycle-GAN](#Cycle-GAN)
* [智能文本处理（PaddleNLP）](#PaddleNLP)
  * [PaddingRNN](#PaddingRNN)
  * [BERT](#BERT)
  * [Transformer](#Transformer)
* [强化学习（PARL）](#PARL)
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

  注意：测试所用GPU服务器为虚拟机，跟相同配置的物理机测试结果可能会有一定的差别。

- CPU服务器参数
  - CPU型号：Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz，24核
  - 指令集：AVX2

## PaddleCV

| 方向 | 模型 | Paddle | TensorFlow | PyTorch | MXNet | 数据集 | batch_size(单卡) |
|---|---|---|---|---|---|---|---|
| 图像分类 | SE-ResNeXt50 | [PaddleCV/image_classification](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification) | - | [SENet-PyTorch](https://github.com/miraclewkf/SENet-PyTorch) | - | ILSVRC2012 | 32 |
| 目标检测 | Mask-RCNN | [PaddleCV/rcnn](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/rcnn) | - | [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) | - | COCO17 | 1 |
| 目标检测 | YOLOv3 | [PaddleCV/yolov3](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/yolov3) | - | - | [gluon-cv](https://github.com/dmlc/gluon-cv/tree/master/scripts/detection/yolo) | COCO17 | 8 |
| 图像分割 | DeepLab V3+ | [PaddleCV/deeplabv3+](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/deeplabv3%2B) | [tensorflow/models](https://github.com/tensorflow/models/tree/master/research/deeplab) | - | - | cityscape | 2 |
| 图像生成 | Cycle-GAN | [PaddleCV/PaddleGAN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleGAN/cycle_gan) | [CycleGAN](https://github.com/hardikbansal/CycleGAN) | - | - | horse2zebra | 1 |

### SE-ResNeXt50
SE-ResNeXt50模型单卡训练速度与PyTorch**持平**，八卡训练速度和显存占用都**优于**PyTorch。

- 训练速度（单位：images/s）

<table>
  <tr>
    <td></td>
    <td colspan=2 align=center><b>CUDA 9.0</b></td>
    <td colspan=2 align=center><b>CUDA 10.0</b></td>
  </tr>
  <tr>
    <td></td>
    <td>Paddle 1.5.0</td>
    <td>PyTorch 1.1.0</td>
    <td>Paddle 1.5.0</td>
    <td>PyTorch 1.1.0</td>
  </tr>
  <tr>
    <td>1 GPU</td>
    <td>168.334</td>
    <td>163.130</td>
    <td>168.478</td>
    <td>163.294</td>
  </tr>
  <tr>
    <td>8 GPUs (单进程)</td>
    <td>843.348</td>
    <td>595.274</td>
    <td>836.357</td>
    <td>573.732</td>
  </tr>
  <tr>
    <td>8 GPUs (多进程)</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
</table>

- 显存占用

<table>
  <tr>
    <td></td>
    <td colspan=2 align=center><b>CUDA 9.0</b></td>
    <td colspan=2 align=center><b>CUDA 10.0</b></td>
  </tr>
  <tr>
    <td></td>
    <td>Paddle 1.5.0</td>
    <td>PyTorch 1.1.0</td>
    <td>Paddle 1.5.0</td>
    <td>PyTorch 1.1.0</td>
  </tr>
  <tr>
    <td>单卡显存占用</td>
    <td>5515 MiB</td>
    <td>5677 MiB</td>
    <td>5535 MiB</td>
    <td>5695 MiB</td>
  </tr>
  <tr>
    <td>单卡最大batch_size</td>
    <td>112</td>
	<td>112</td>
	<td>112</td>
	<td>112</td>
  </tr>
</table>

### Mask-RCNN
Mask-RCNN模型训练速度和显存占用都**优于**PyTorch。

- 训练速度（单位：images/s）

<table>
  <tr>
    <td></td>
    <td colspan=2 align=center><b>CUDA 9.0</b></td>
    <td colspan=2 align=center><b>CUDA 10.0</b></td>
  </tr>
  <tr>
    <td></td>
    <td>Paddle 1.5.0</td>
    <td>PyTorch 1.1.0</td>
    <td>Paddle 1.5.0</td>
    <td>PyTorch 1.1.0</td>
  </tr>
  <tr>
    <td>1 GPU</td>
    <td>3.811</td>
    <td>3.240</td>
    <td>3.780</td>
    <td>-</td>
  </tr>
  <tr>
    <td>8 GPUs (单进程)</td>
    <td>18.707</td>
    <td>-</td>
    <td>18.505</td>
    <td>-</td>
  </tr>
  <tr>
    <td>8 GPUs (多进程)</td>
    <td>23.014</td>
    <td>21.864</td>
    <td>23.199</td>
    <td>-</td>
  </tr>
</table>

- 显存占用

<table>
  <tr>
    <td></td>
    <td colspan=2 align=center><b>CUDA 9.0</b></td>
    <td colspan=2 align=center><b>CUDA 10.0</b></td>
  </tr>
  <tr>
    <td></td>
    <td>Paddle 1.5.0</td>
    <td>PyTorch 1.1.0</td>
    <td>Paddle 1.5.0</td>
    <td>PyTorch 1.1.0</td>
  </tr>
  <tr>
    <td>单卡显存占用</td>
    <td>3871 MiB</td>
    <td>4548 MiB</td>
    <td>3907 MiB</td>
    <td>-</td>
  </tr>
  <tr>
    <td>单卡最大batch_size</td>
    <td>5</td>
    <td>5</td>
    <td>5</td>
    <td>-</td>
  </tr>
</table>

### YOLOv3
YOLOv3模型训练速度和显存占用都**优于**MXNet。

- 训练速度（单位：images/s）

<table>
  <tr>
    <td></td>
    <td colspan=2 align=center><b>CUDA 9.0</b></td>
    <td colspan=2 align=center><b>CUDA 10.0</b></td>
  </tr>
  <tr>
    <td></td>
    <td>Paddle 1.5.0</td>
    <td>MXNet</td>
    <td>Paddle 1.5.0</td>
    <td>MXNet</td>
  </tr>
  <tr>
    <td>1 GPU</td>
    <td>29.901</td>
    <td>18.578</td>
    <td>30.591</td>
    <td>17.001</td>
  </tr>
  <tr>
    <td>8 GPUs (单进程)</td>
    <td>58.175</td>
    <td>35.574</td>
    <td>57.997</td>
    <td>33.755</td>
  </tr>
  <tr>
    <td>8 GPUs (多进程)</td>
    <td>99.530</td>
    <td>-</td>
    <td>104.553</td>
    <td>-</td>
  </tr>
</table>

- 显存占用

<table>
  <tr>
    <td></td>
    <td colspan=2 align=center><b>CUDA 9.0</b></td>
    <td colspan=2 align=center><b>CUDA 10.0</b></td>
  </tr>
  <tr>
    <td></td>
    <td>Paddle 1.5.0</td>
    <td>MXNet</td>
    <td>Paddle 1.5.0</td>
    <td>MXNet</td>
  </tr>
  <tr>
    <td>单卡显存占用</td>
    <td>10583 MiB</td>
    <td>14304 MiB</td>
    <td>10599 MiB</td>
    <td>9842 MiB</td>
  </tr>
  <tr>
    <td>单卡最大batch_size</td>
    <td>14</td>
	<td>14</td>
	<td>14</td>
	<td>14</td>
  </tr>
</table>

### DeepLab V3+
Deep Lab V3+模型训练速度和显存占用都**优于**TensorFlow。

- 训练速度（单位：images/s）

<table>
  <tr>
    <td></td>
    <td colspan=2 align=center><b>CUDA 9.0</b></td>
    <td colspan=2 align=center><b>CUDA 10.0</b></td>
  </tr>
  <tr>
    <td></td>
    <td>Paddle 1.5.0</td>
    <td>TensorFlow 1.12.0</td>
    <td>Paddle 1.5.0</td>
    <td>TensorFlow 1.14.0</td>
  </tr>
  <tr>
    <td>1 GPU</td>
    <td>13.695</td>
    <td>6.4</td>
    <td>14.261</td>
    <td>6.309</td>
  </tr>
  <tr>
    <td>8 GPUs (单进程)</td>
    <td>59.721</td>
    <td>16.508</td>
    <td>58.024</td>
    <td>16.427</td>
  </tr>
</table>

- 显存占用

<table>
  <tr>
    <td></td>
    <td colspan=2 align=center><b>CUDA 9.0</b></td>
    <td colspan=2 align=center><b>CUDA 10.0</b></td>
  </tr>
  <tr>
    <td></td>
    <td>Paddle 1.5.0</td>
    <td>TensorFlow 1.12.0</td>
    <td>Paddle 1.5.0</td>
    <td>TensorFlow 1.14.0</td>
  </tr>
  <tr>
    <td>单卡显存占用</td>
    <td>5163 MiB</td>
    <td>8934 MiB</td>
    <td>5167 MiB</td>
    <td>8927 MiB</td>
  </tr>
  <tr>
    <td>单卡最大batch_size</td>
    <td>9</td>
    <td>7</td>
    <td>9</td>
    <td>7</td>
  </tr>
</table>

### Cycle-GAN
Cycle-GAN模型不支持多卡训练，其单卡训练速度和显存占用都**优于**TensorFlow。

- 训练速度（单位：images/s）

<table>
  <tr>
    <td></td>
    <td colspan=2 align=center><b>CUDA 9.0</b></td>
    <td colspan=2 align=center><b>CUDA 10.0</b></td>
  </tr>
  <tr>
    <td></td>
    <td>Paddle 1.5.0</td>
    <td>TensorFlow 1.12.0</td>
    <td>Paddle 1.5.0</td>
    <td>TensorFlow 1.14.0</td>
  </tr>
  <tr>
    <td>1 GPU</td>
    <td>7.513</td>
    <td>6.452</td>
    <td>7.591</td>
    <td>6.823</td>
  </tr>
</table>

- 显存占用

<table>
  <tr>
    <td></td>
    <td colspan=2 align=center><b>CUDA 9.0</b></td>
    <td colspan=2 align=center><b>CUDA 10.0</b></td>
  </tr>
  <tr>
    <td></td>
    <td>Paddle 1.5.0</td>
    <td>TensorFlow 1.12.0</td>
    <td>Paddle 1.5.0</td>
    <td>TensorFlow 1.14.0</td>
  </tr>
  <tr>
    <td>单卡显存占用</td>
    <td>2479 MiB</td>
    <td>5094 MiB</td>
    <td>2499 MiB</td>
    <td>5089 MiB</td>
  </tr>
</table>

## PaddleNLP

| 方向 | 模型 | Paddle | TensorFlow | PyTorch | 数据集 | batch_size(单卡) |
|---|---|---|---|---|---|---|
| 语言模型 | PaddingRNN | [PaddleNLP/language_model](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/language_model) | [benchmark/PaddingRNN/lstm_tf](https://github.com/PaddlePaddle/benchmark/tree/master/PaddingRNN/lstm_tf) | - | PTB文本数据集 | 20 |
| 语义表示 | BERT | [LARK](https://github.com/PaddlePaddle/LARK) | [google-research/bert](https://github.com/google-research/bert) | - | XNLI | 32 |
| 机器翻译 | Transformer | [PaddleNLP/neural_machine_translation/transformer](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/neural_machine_translation/transformer) | [tensor2tensor](https://github.com/tensorflow/tensor2tensor) | - | En-de | 4096 |

### PaddingRNN
TensorFlow的PaddingRNN开源模型多卡训练失败，故只测试单卡训练的情况。
PaddleRNN模型在static模式下，单卡训练速度和显存占用都**差于**TensorFlow。

- 训练速度（单位：steps/s）

<table>
  <tr>
    <td rowspan=3><b>static模式<br>small模型</b></td>
    <td></td>
    <td colspan=2 align=center><b>CUDA 9.0</b></td>
    <td colspan=2 align=center><b>CUDA 10.0</b></td>
  </tr>
  <tr>
    <td></td>
    <td>Paddle 1.5.0</td>
    <td>TensorFlow 1.12.0</td>
    <td>Paddle 1.5.0</td>
    <td>TensorFlow 1.14.0</td>
  </tr>
  <tr>
    <td>1 GPU</td>
    <td>61.208</td>
    <td>73.991</td>
    <td>63.400</td>
    <td>72.406</td>
  </tr>
</table>

<table>
  <tr>
    <td rowspan=3><b>static模式<br>large模型</b></td>
    <td></td>
    <td colspan=2 align=center><b>CUDA 9.0</b></td>
    <td colspan=2 align=center><b>CUDA 10.0</b></td>
  </tr>
  <tr>
    <td></td>
    <td>Paddle 1.5.0</td>
    <td>TensorFlow 1.12.0</td>
    <td>Paddle 1.5.0</td>
    <td>TensorFlow 1.14.0</td>
  </tr>
  <tr>
    <td>1 GPU</td>
    <td>17.479</td>
    <td>18.529</td>
    <td>17.107</td>
    <td>17.914</td>
  </tr>
</table>

- 显存占用

<table>
  <tr>
    <td rowspan=3><b>static模式<br>small模型</b></td>
    <td></td>
    <td colspan=2 align=center><b>CUDA 9.0</b></td>
    <td colspan=2 align=center><b>CUDA 10.0</b></td>
  </tr>
  <tr>
    <td></td>
    <td>Paddle 1.5.0</td>
    <td>TensorFlow 1.12.0</td>
    <td>Paddle 1.5.0</td>
    <td>TensorFlow 1.14.0</td>
  </tr>
  <tr>
    <td>单卡显存占用</td>
    <td>660 MiB</td>
    <td>660 MiB</td>
    <td>657 MiB</td>
    <td>647 MiB</td>
  </tr>
</table>

<table>
  <tr>
    <td rowspan=3><b>static模式<br>large模型</b></td>
    <td></td>
    <td colspan=2 align=center><b>CUDA 9.0</b></td>
    <td colspan=2 align=center><b>CUDA 10.0</b></td>
  </tr>
  <tr>
    <td></td>
    <td>Paddle 1.5.0</td>
    <td>TensorFlow 1.12.0</td>
    <td>Paddle 1.5.0</td>
    <td>TensorFlow 1.14.0</td>
  </tr>
  <tr>
    <td>单卡显存占用</td>
    <td>6089 MiB</td>
    <td>5858 MiB</td>
    <td>6083 MiB</td>
    <td>8711 MiB</td>
  </tr>
</table>

### BERT
TensorFlow的BERT开源模型暂无多卡实现。
BERT模型单卡训练速度和显存占用都优于TensorFlow。

- 训练速度（单位：steps/s）

<table>
  <tr>
    <td></td>
    <td colspan=2 align=center><b>CUDA 9.0</b></td>
    <td colspan=2 align=center><b>CUDA 10.0</b></td>
  </tr>
  <tr>
    <td></td>
    <td>Paddle 1.5.0</td>
    <td>TensorFlow 1.12.0</td>
    <td>Paddle 1.5.0</td>
    <td>TensorFlow 1.14.0</td>
  </tr>
  <tr>
    <td>1 GPU</td>
    <td>4.044</td>
    <td>3.420</td>
    <td>4.003</td>
    <td>-</td>
  </tr>
  <tr>
    <td>8 GPUs (单进程)</td>
    <td>1.803</td>
    <td>-</td>
    <td>1.817</td>
    <td>-</td>
  </tr>
  <tr>
    <td>8 GPUs (多进程)</td>
    <td>3.114</td>
    <td>-</td>
    <td>3.089</td>
    <td>-</td>
  </tr>
</table>

- 显存占用

<table>
  <tr>
    <td></td>
    <td colspan=2 align=center><b>CUDA 9.0</b></td>
    <td colspan=2 align=center><b>CUDA 10.0</b></td>
  </tr>
  <tr>
    <td></td>
    <td>Paddle 1.5.0</td>
    <td>TensorFlow 1.12.0</td>
    <td>Paddle 1.5.0</td>
    <td>TensorFlow 1.14.0</td>
  </tr>
  <tr>
    <td>单卡显存占用</td>
    <td>6551 MiB</td>
    <td>15430 MiB</td>
    <td>6545 MiB</td>
    <td>-</td>
  </tr>
  <tr>
    <td>单卡最大batch_size</td>
    <td>9984</td>
    <td>9216</td>
    <td>9984</td>
    <td>-</td>
  </tr>
</table>

### Transformer
Transformer模型单卡训练速度与TensorFlow**持平**；多卡训练速度和显存占用**优于**TensorFlow。

- 训练速度（单位：steps/s）

<table>
  <tr>
    <td></td>
    <td colspan=2 align=center><b>CUDA 9.0</b></td>
    <td colspan=2 align=center><b>CUDA 10.0</b></td>
  </tr>
  <tr>
    <td></td>
    <td>Paddle 1.5.0</td>
    <td>TensorFlow 1.12.0</td>
    <td>Paddle 1.5.0</td>
    <td>TensorFlow 1.14.0</td>
  </tr>
  <tr>
    <td>1 GPU</td>
    <td>4.865</td>
    <td>4.750</td>
    <td>4.883</td>
    <td>4.721</td>
  </tr>
  <tr>
    <td>8 GPUs (单进程)</td>
    <td>4.227</td>
    <td>2.302</td>
    <td>4.355</td>
    <td>2.520</td>
  </tr>
  <tr>
    <td>8 GPUs (多进程)</td>
    <td>4.445</td>
    <td>-</td>
    <td>4.460</td>
    <td>-</td>
  </tr>
</table>

- 显存占用

<table>
  <tr>
    <td></td>
    <td colspan=2 align=center><b>CUDA 9.0</b></td>
    <td colspan=2 align=center><b>CUDA 10.0</b></td>
  </tr>
  <tr>
    <td></td>
    <td>Paddle 1.5.0</td>
    <td>TensorFlow 1.12.0</td>
    <td>Paddle 1.5.0</td>
    <td>TensorFlow 1.14.0</td>
  </tr>
  <tr>
    <td>单卡显存占用</td>
    <td>7137 MiB</td>
    <td>8948 MiB</td>
    <td>7147 MiB</td>
    <td>8711 MiB</td>
  </tr>
  <tr>
    <td>单卡最大batch_size</td>
    <td>12000</td>
    <td>11144</td>
    <td>12000</td>
    <td>11144</td>
  </tr>
</table>

## PARL

| 方向 | 模型 | Paddle | TensorFlow | PyTorch | 数据集 | batch_size(单卡) |
|---|---|---|---|---|---|---|
| 强化学习 | DDPG Deep Explore |	[benchmark/DDPG_Deep_Explore/Fluid_version](https://github.com/PaddlePaddle/benchmark/tree/master/DDPG_Deep_Explore/Fluid_version) | [benchmark/DDPG_Deep_Explore/TF_version](https://github.com/PaddlePaddle/benchmark/tree/master/DDPG_Deep_Explore/TF_version) | - | 测试数据 | 100 |

### DDPG
DDPG模型不支持多卡训练，其训练速度与竞品持平，显存占用**优于**TensorFlow。

- 训练速度（单位：epoch/s）

<table>
  <tr>
    <td></td>
    <td colspan=2 align=center><b>CUDA 9.0</b></td>
    <td colspan=2 align=center><b>CUDA 10.0</b></td>
  </tr>
  <tr>
    <td></td>
    <td>Paddle 1.5.0</td>
    <td>TensorFlow 1.12.0</td>
    <td>Paddle 1.5.0</td>
    <td>TensorFlow 1.14.0</td>
  </tr>
  <tr>
    <td>1 GPU</td>
    <td>1.615</td>
    <td>1.606</td>
    <td>1.578</td>
    <td>-</td>
  </tr>
</table>

- 显存占用

<table>
  <tr>
    <td></td>
    <td colspan=2 align=center><b>CUDA 9.0</b></td>
    <td colspan=2 align=center><b>CUDA 10.0</b></td>
  </tr>
  <tr>
    <td></td>
    <td>Paddle 1.5.0</td>
    <td>TensorFlow 1.12.0</td>
    <td>Paddle 1.5.0</td>
    <td>TensorFlow 1.14.0</td>
  </tr>
  <tr>
    <td>单卡显存占用</td>
    <td>563 MiB</td>
    <td>630 MiB</td>
    <td>557 MiB</td>
    <td>-</td>
  </tr>
</table>