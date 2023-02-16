#!/usr/bin/env bash
# 执行路径在模型库的根目录下
################################# 安装框架 如:
wget -nc https://paddle-wheel.bj.bcebos.com/benchmark/torch-1.8.1%2Bcu111-cp37-cp37m-linux_x86_64.whl
python -m pip install torch-1.8.1+cu111-cp37-cp37m-linux_x86_64.whl
python -m pip install torchvision==0.9.1 torchaudio==0.8.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -r requirement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ./mmdetection-2.11.0
pip install -v -e .
cd ..
pip install -v -e .

wget https://paddleocr.bj.bcebos.com/dataset/mmcv-1.3.4-py2.py3-none-any.whl
pip install mmcv-1.3.4-py2.py3-none-any.whl
# 下载数据集并解压
wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dataset/StructureLabel_val_500.tar --no-check-certificate
cd ./train_data/ && tar xf StructureLabel_val_500.tar
cd ../

export MASTER_ADDR="localhost"
export MASTER_PORT="6006"
