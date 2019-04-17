# Distributed Training Benchmark For PaddlePaddle

We release distributed training benchmark in this repository. The following tasks will be included for user reference.

## 1.Parameter Server Based Training

### 1.1 Click Through Rate Estimation

### 1.2 Word2vec

### 1.3 Simnet-Bow

## 2. Collective Training

### 2.1 Resnet50

#### 2.1.1 Repo
[image_classification](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification/dist_train)

#### 2.1.2 Performance

The below figure shows fluid distributed training performances. We did these on a 4-node V100 GPU cluster,
each has 8 V100 GPU card, with total of 32 GPUs. All modes can reach the "state of the art (choose loss scale carefully when using fp16 mode)" of ResNet50 model with imagenet dataset. The Y axis in the figure shows
the images/s while the X-axis shows the number of GPUs.

<p align="center">
<img src="https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/images/imagenet_dist_performance.png?raw=true" width=528> <br />
Performance of Multiple-GPU Training of Resnet50 on Imagenet
</p>

The second figure shows speed-ups when using multiple GPUs according to the above figure.

<p align="center">
<img src="https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/images/imagenet_dist_speedup.png?raw=true" width=528> <br />
Speed-ups of Multiple-GPU Training of Resnet50 on Imagenet
</p>


### 2.1.3 Deep Gradient Compression([arXiv:1712.01887](https://arxiv.org/abs/1712.01887)) for resnet

#### Environment

  - GPU: NVIDIA® Tesla® V100 
  - Machine number * Card number: 4 * 4
  - System: Centos 6u3
  - Cuda/Cudnn: 9.0/7.1
  - Dataset: ImageNet
  - Date: 2017.04
  - PaddleVersion: 1.4
  - Batch size: 32

#### Performance

<p align="center">
<img src="https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/images/resnet_dgc.png?raw=true" width=528> <br />
Performance using DGC for resnet-fp32 under different bandwidth
</p>


### 2.2 Se-Resnet50

### 2.3 Transformer

### 2.4 Bert

### 2.5 VGG16

