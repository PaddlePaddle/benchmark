# Modulus(pytorch后端) 动态图 benchmark 模型执行说明

动态图benchmark测试脚本说明

## 1. 目录说明

- models/modulus 文件夹为剥离符号化相关代码的核心代码库
- models/modulus-sym 文件夹为基本只包含符号计算相关的代码库

modulus 官方套件按照上述文件分离方式进行设计，并且使用单一的 pytorch 作为后端。

## 2. Docker 运行环境

docker image: registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.8-cudnn8.6-trt8.5-gcc82
paddle = 0.0.0
python >= 3.9

## 3. 运行 benchmark 测试步骤

以 annular_ring-annular_ring_equation_instancing-annular_ring 模型为例，运行命令如下。

``` sh
export ROOT_DIR=$PWD
cd ${ROOT_DIR}

git clone https://github.com/PaddlePaddle/benchmark.git -b develop
wget https://paddle-qa.bj.bcebos.com/benchmark/tools.tar.gz
export BENCHMARK_ROOT=${ROOT_DIR}/benchmark-frame/tools

model_path=${ROOT_DIR}/benchmark/frame_benchmark/pytorch/dynamic/modulus-sym/models/modulus-sym
script_path=${ROOT_DIR}benchmark/frame_benchmark/pytorch/dynamic/modulus-sym/scripts/annular_ring-annular_ring_equation_instancing-annular_ring/N1C1/annular_ring-annular_ring_equation_instancing-annular_ring_bs1_fp32_DP.sh

\cp ${script_path} ${model_path}
model_benchmark_common=$(dirname $(dirname ${script_path}))
\cp -r ${model_benchmark_common}/benchmark_common/* ${model_path}/

cd ${model_path}
bash annular_ring-annular_ring_equation_instancing-annular_ring_bs1_fp32_DP.sh
```
