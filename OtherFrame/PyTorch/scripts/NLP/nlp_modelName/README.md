<!-- omit in toc -->
# NGC PyTorch Bert 性能复现
## 脚本说明
└── nlp_modelName              # 模型名  
    ├── README.md              # 运行文档  
    ├── analysis_log.py        # log解析脚本,每个框架尽量统一   
    ├── logs                   # 训练log,注:log中不得包含机器ip等敏感信息  
    │   ├── index              # log解析后待入库数据json文件   
    │   │   ├── nlp_modelName_sp_bs32_fp32_1_speed  # 单卡数据  
    │   │   └── nlp_modelName_sp_bs32_fp32_8_speed  # 8卡数据  
    │   └── train_log          # 原始训练log  
    ├── preData.sh             # 数据处理  
    └── run_benchmark.sh       # 运行脚本（包含性能、收敛性）  


此处给出了基于 [NGC PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT) 实现的 Bert Base Pre-Training 任务的详细复现流程，包括执行环境、PyTorch版本、环境搭建、复现脚本、测试结果和测试日志。

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

物理机环境，对 [NGC PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT) 的 Bert 模型进行了测试，详细物理机配置，见[Paddle Bert Base 性能测试](../../README.md#1.物理机环境)。

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

NGC PyTorch 的代码仓库提供了自动构建 Docker 镜像的的 [shell 脚本](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT/scripts/docker/build.sh)，

- **镜像版本**: `nvcr.io/nvidia/pytorch:20.06-py3`
- **PyTorch 版本**: `1.6.0a0+9907a3e`
- **CUDA 版本**: `11.0.167`
- **cuDnn 版本**: `8.0.1`

## 二、环境搭建

### 1. 单机（单卡、8卡）环境搭建

我们遵循了 NGC PyTorch 官网提供的 [Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#quick-start-guide) 教程搭建了测试环境，主要过程如下：

- **拉取代码**

    ```bash
    git clone https://github.com/NVIDIA/DeepLearningExamples
    cd DeepLearningExamples/PyTorch/LanguageModeling/BERT
    # 本次测试是在如下版本下完成的：
    git checkout 8d8c524df634e4dfa0cfbf77a904ce2ede85e2ec
    ```
- **准备数据** （也可写到数据处理脚本中,需提供小数据集,训练时间在5min内）

    NGC PyTorch 提供单独的数据下载和预处理脚本 [data/create_datasets_from_start.sh](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/data/create_datasets_from_start.sh)。在容器中执行如下命令，可以下载和制作 `wikicorpus_en` 的 hdf5 数据集。

    ```bash
    bash data/create_datasets_from_start.sh wiki_only
    ```

    由于数据集比较大，且容易受网速的影响，上述命令执行时间较长。因此，为了更方便复现竞品的性能数据，我们提供了已经处理好的 seq_len=128 的 hdf5 格式[样本数据集](https://bert-data.bj.bcebos.com/benchmark_sample%2Fhdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5.tar.gz)，共100个 part hdf5 数据文件，约 3.1G。

    数据下载后，会得到一个 `hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5.tar.gz`压缩文件：

    ```bash
    # 解压数据集
    tar -xzvf benchmark_sample_hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5.tar.gz

    # 放到 data/ 目录下
    mv benchmark_sample_hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5 bert/data/
    ```

    修改 [scripts/run_pretraining.sh](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/scripts/run_pretraining.sh#L37)脚本的 `DATASET`变量为上述数据集地址即可。


## 三、测试步骤
为了更方便地测试不同 batch_size、num_gpus 组合下的性能，我们单独编写了 `run_benchmark.sh` 脚本,该脚本需包含运行模型的命令和分析log产出待入库json文件的脚本,详细内容请见脚本;

**重要的配置参数：**
- **run_mode**: 单卡sp|多卡mp
- **batch_size**: 用于第一阶段的单卡总 batch_size
- **fp_item**: 用于指定精度训练模式，fp32 或 fp16 
- **max_iter**: 运行的最大iter或epoch,根据模型选择

- **gradient_accumulation_steps**: 每次执行 optimizer 前的梯度累加步数
- **BERT_CONFIG:** 用于指定 base 或 large 模型的参数配置文件 (line:49)
- **bert_model:** 用于指定模型类型，默认为`bert-large-uncased`

### 1. 单机（单卡、8卡）测试
- **单卡启动脚本：**

    若测试单机单卡 batch_size=32、FP32 的训练性能，执行如下命令：

    ```bash
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 32 fp32 500 ${log_path}   # 如果fp32\fp16不方便放在一个脚本,可另写
    ```

- **8卡启动脚本：**

    若测试单机8卡 batch_size=64、FP16 的训练性能，执行如下命令：

    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 64 fp16 500   
    ```
- **收敛性验证：**

    若测试单机8卡 batch_size=64、FP16 的收敛性，执行如下命令：

    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 64 fp16 50000  
    ```

## 四、测试结果
### 速度结果 
> 单位： sequences/sec  
  
|卡数 | FP32(BS=32) | FP32(BS=48) | AMP(BS=64) | AMP(BS=96)|
|:-----:|:-----:|:-----:|:-----:|:-----:|
|1 | 127.09  | 126.48  | 511.13  | 532.24  |
|8 | 1018.40  | 1012.80  | 4063.19  | 4207.36 |
> 特殊情况说明,如:关于batch_size 从32增加到48时，8卡和32卡性能并没有提升的问题，我们反复重测了多次。若了解相关原因，欢迎issue我们。

### 显存结果 
> 单位： MB 
  
|卡数 | FP32(BS=32) | FP32(BS=48) | AMP(BS=64) | AMP(BS=96)|
|:-----:|:-----:|:-----:|:-----:|:-----:|
|1 |   |   |   |   |
|8 |   |  |   |  |
> 特殊情况说明,如:关于batch_size 从32增加到48时，8卡和32卡性能并没有提升的问题，我们反复重测了多次。若了解相关原因，欢迎issue我们。


## 五、日志数据
### 1.单机（单卡、8卡）日志

- [单卡 bs=32、FP32](./logs/bert_base_lamb_pretraining.pyt_bert_pretraining_phase1_fp32_bs32_gpu1.log)
- [单卡 bs=48、FP32](./logs/bert_base_lamb_pretraining.pyt_bert_pretraining_phase1_fp32_bs48_gpu1.log)
- [单卡 bs=64、AMP](./logs/bert_base_lamb_pretraining.pyt_bert_pretraining_phase1_fp16_bs64_gpu1.log)
- [单卡 bs=96、AMP](./logs/bert_base_lamb_pretraining.pyt_bert_pretraining_phase1_fp16_bs96_gpu1.log)
- [8卡 bs=32、FP32](./logs/bert_base_lamb_pretraining.pyt_bert_pretraining_phase1_fp32_bs32_gpu8.log)
- [8卡 bs=48、FP32](./logs/bert_base_lamb_pretraining.pyt_bert_pretraining_phase1_fp32_bs48_gpu8.log)
- [8卡 bs=64、AMP](./logs/bert_base_lamb_pretraining.pyt_bert_pretraining_phase1_fp16_bs64_gpu8.log)
- [8卡 bs=96、AMP](./logs/bert_base_lamb_pretraining.pyt_bert_pretraining_phase1_fp16_bs96_gpu8.log)
- [32卡 bs=32、FP32](./logs/bert_base_lamb_pretraining.pyt_bert_pretraining_phase1_fp32_bs32_gpu32.log)
- [32卡 bs=48、FP32](./logs/bert_base_lamb_pretraining.pyt_bert_pretraining_phase1_fp32_bs48_gpu32.log)
- [32卡 bs=64、AMP](./logs/bert_base_lamb_pretraining.pyt_bert_pretraining_phase1_fp16_bs64_gpu32.log)
- [32卡 bs=96、AMP](./logs/bert_base_lamb_pretraining.pyt_bert_pretraining_phase1_fp16_bs96_gpu32.log)
- [32卡 bs=32、FP32 no GradAcc](./logs/bert_base_lamb_pretraining.pyt_bert_pretraining_phase1_without_gradacc_fp32_bs32_gpu32.log)
- [32卡 bs=48、FP32 no GradAcc](./logs/bert_base_lamb_pretraining.pyt_bert_pretraining_phase1_without_gradacc_fp32_bs48_gpu32.log)
- [32卡 bs=64、AMP  no GradAcc](./logs/bert_base_lamb_pretraining.pyt_bert_pretraining_phase1_without_gradacc_fp16_bs64_gpu32.log)
- [32卡 bs=96、AMP  no GradAcc](./logs/bert_base_lamb_pretraining.pyt_bert_pretraining_phase1_without_gradacc_fp16_bs96_gpu32.log)
 