<!-- omit in toc -->
# PSENet 性能复现

此处给出了基于 PSENet 的详细复现流程，包括执行环境、PyTorch版本、环境搭建、复现脚本、测试结果和测试日志。

<!-- omit in toc -->
## 目录
- [一、环境介绍](#一环境介绍)
  - [1.物理机环境](#1物理机环境)
  - [2.Docker 镜像](#2docker-镜像)
- [二、环境搭建](#二环境搭建)
  - [1. 单机（单卡、8卡）环境搭建](#1-单机单卡8卡环境搭建)
- [三、测试步骤](#三测试步骤)
  - [1. 单机（单卡、8卡）测试](#1-单机单卡8卡测试)
- [四、测试结果](#四测试结果)
- [五、日志数据](#五日志数据)
  - [1.单机（单卡、8卡）日志](#1单机单卡8卡日志)


## 一、环境介绍

### 1.物理机环境（如每个框架用的一致可链接到标准环境）

物理机环境，对 PSENet 模型进行了测试，详细物理机配置如下。

- 单机（单卡、8卡）
  - 系统：CentOS release 7.5 (Final)
  - GPU：Tesla V100-SXM2-16GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz * 38
  - Driver Version: 460.32.03
  - 内存：502 GB
 
- 多机（32卡）
  - 系统：CentOS release 6.3 (Final)
  - GPU：Tesla V100-SXM2-32GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz * 48
  - Driver Version: 450.80.02
  - 内存：502 GB

### 2.Docker 镜像（如每个框架用的一致可链接到标准环境）

- **镜像版本**: `registry.baidubce.com/paddlepaddle/paddle:2.0.2-gpu-cuda10.1-cudnn7`
- **PyTorch 版本**: `1.7.1` 
- **CUDA 版本**: `10.1`
- **cuDnn 版本**: `7`

## 二、环境搭建

### 1. 单机（单卡、8卡）环境搭建

我们遵循了 PSENet [官方repo](https://github.com/whai362/PSENet) 的教程搭建了测试环境，主要过程如下：

- **拉取代码**

    ```bash
    cd OtherFrame/ocr/PyTorch/models
    git clone https://github.com/WenmuZhou/PSENet.git
    cd PSENet
    # 本次测试是在如下版本下完成的：
    git checkout ff593281bfe81721a6f1f26b17a0958f017f051e
    ```
- **准备数据** （也可写到数据处理脚本中,需提供小数据集,训练时间在5min内）

    ```bash
    cp ../../scripts/PSENet/preData.sh ./;
    bash preData.sh
    ```

    数据下载后，在当前目录下生成 `train_data/icdar2015` 文件夹，里面存放了图片和label：


## 三、测试步骤
为了更方便地测试不同 batch_size、num_gpus 组合下的性能，我们单独编写了 `run_benchmark.sh` 脚本,该脚本需包含运行模型的命令和分析log产出待入库json文件的脚本,详细内容请见脚本;

**重要的配置参数：**
- **run_mode**: 单卡sp|多卡mp
- **batch_size**: 用于第一阶段的单卡总 batch_size
- **fp_item**: 用于指定精度训练模式，fp32
- **max_iter**: 运行的最大iter或epoch,根据模型选择，PSENet设置为2

### 1. 单机（单卡、8卡）测试
- **单卡启动脚本：**

    若测试单机单卡 batch_size=32、FP32 的训练性能，执行如下命令：

    ```bash
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 8 fp32 2 ${log_path}   # 如果fp32\fp16不方便放在一个脚本,可另写
    ```

- **8卡启动脚本：**

    若测试单机8卡 batch_size=64、FP16 的训练性能，执行如下命令：

    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 64 fp32 2
    ```
- **收敛性验证：**

    若测试单机8卡 batch_size=64、FP16 的收敛性，执行如下命令，收敛指标：（如loss10.+下降到0.05 附近，收敛耗时：v100*32G*1卡*14h）

    ```bash
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 16 fp32 600
    ```
