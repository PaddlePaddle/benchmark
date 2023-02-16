#!/usr/bin/env bash
# 执行路径在模型库的根目录下
################################# 安装框架 如:
# wget -nc ${FLAG_TORCH_WHL_URL}
# tar -xvf torch_dev_whls.tar
# pip install torch_dev_whls/*

cd ./mmdetection-2.11.0
pip install -v -e .
cd ..
pip install -v -e .

wget  -nc https://paddleocr.bj.bcebos.com/dataset/mmcv-1.3.4-py2.py3-none-any.whl
pip install mmcv-1.3.4-py2.py3-none-any.whl
# 下载数据集并解压
wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dataset/StructureLabel_val_500.tar --no-check-certificate
cd ./train_data/ && tar xf StructureLabel_val_500.tar
cd ../

export MASTER_ADDR="localhost"
export MASTER_PORT="6006"
