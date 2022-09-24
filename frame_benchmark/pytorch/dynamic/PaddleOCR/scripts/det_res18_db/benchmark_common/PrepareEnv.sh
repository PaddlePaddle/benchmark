#!/usr/bin/env bash
# 执行路径在模型库的根目录下
################################# 安装框架 如:
wget -nc https://paddle-wheel.bj.bcebos.com/benchmark/torch-1.8.1%2Bcu111-cp37-cp37m-linux_x86_64.whl
python -m pip install torch-1.8.1+cu111-cp37-cp37m-linux_x86_64.whl
python -m pip install torchvision==0.9.1 torchaudio==0.8.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -r requirement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 下载数据集并解压
rm -rf datasets
wget -nc https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/benchmark_train/datasets.tar
tar xf datasets.tar

export MASTER_ADDR="localhost"
export MASTER_PORT="6006"