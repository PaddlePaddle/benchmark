# Modulus(pytorch后端) 动态图 benchmark 模型执行说明

动态图benchmark测试脚本说明

## 1. 目录说明

- models/modulus 文件夹为剥离符号化相关代码的核心代码库
- models/modulus-sym 文件夹为基本只包含符号计算相关的代码库

modulus 官方套件按照上述文件分离方式进行设计，并且使用单一的 pytorch 作为后端。

## 2. Docker 运行环境

docker image: registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.2-cudnn8-gcc82
paddle = 0.0.0
python = 3.7

## 3. 运行benchmark测试步骤

git clone <https://github.com/PaddleBenchmark/deepxde> -b pytorch_benchmark
cd deepxde

## 4. 运行模型

### 4.1 单卡（自动运行打开Profiling）

export CUDA_VISIBLE_DEVICES=0
bash scripts/Eular_beam/N1C1/Eular_beam_bs1_fp32_DP.sh
bash scripts/Navier_Stokes_inverse/N1C1/Navier_Stokes_inverse_bs1_fp32_DP.sh

### 4.2 多卡（暂无需求）

### 4.3 profiling（无）
