#!/usr/bin/env bash
# 执行路径在模型库的根目录下
################################# 安装框架 如:
wget -nc ${FLAG_TORCH_WHL_URL}
tar -xvf torch_dev_whls.tar
python -m pip install torch_dev_whls/*
python -m pip install -r requirement.txt
# 下载数据集并解压
rm -rf datasets
wget -nc https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/benchmark_train/datasets.tar
tar xf datasets.tar

# Download resnet50 checkpoint
wget -nc https://paddle-wheel.bj.bcebos.com/benchmark/resnet50-19c8e357.pth -O /root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth

export MASTER_ADDR="localhost"
export MASTER_PORT="6006"
