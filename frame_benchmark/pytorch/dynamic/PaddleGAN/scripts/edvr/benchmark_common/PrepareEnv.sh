#!/usr/bin/env bash

# install env
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
echo "*******prepare benchmark start ***********"
pip install -U pip
echo `pip --version`


pip install https://paddle-wheel.bj.bcebos.com/benchmark/torch-1.12.0%2Bcu113-cp37-cp37m-linux_x86_64.whl
pip install https://paddle-wheel.bj.bcebos.com/benchmark/torchvision-0.13.0%2Bcu113-cp37-cp37m-linux_x86_64.whl
pip install setuptools==50.3.2
pip install -v -e .


# mmcv-full wheel takes too long to compile online,
# however this wheel relies on compile environment
# different compiled wheels are provided for different cluster
if [ `nvidia-smi --list-gpus | grep A100 | wc -l` -ne "0" ]; then
    echo "Run on A100 Cluster"
    wget https://paddle-wheel.bj.bcebos.com/benchmark/mmcv_full-1.5.0-cp37-cp37m-linux_x86_64_A100.whl -O mmcv_full-1.5.0-cp37-cp37m-linux_x86_64.whl
else
    echo "Run on V100 Cluster"
    wget https://paddle-wheel.bj.bcebos.com/benchmark/mmcv_full-1.5.0-cp37-cp37m-linux_x86_64_V100.whl -O mmcv_full-1.5.0-cp37-cp37m-linux_x86_64.whl
fi
pip install mmcv_full-1.5.0-cp37-cp37m-linux_x86_64.whl && rm -f mmcv_full-1.5.0-cp37-cp37m-linux_x86_64.whl


################################# 准备训练数据 如:
mkdir -p data/REDS
wget https://paddle-wheel.bj.bcebos.com/benchmark/REDS_small.zip -O data/REDS_small.zip
unzip -o data/REDS_small.zip
mv REDS_small/* data/REDS/ && rm -rf REDS_small.zip REDS_small
echo "download data" #waiting data process
echo "dataset prepared done"
echo "*******prepare benchmark end***********"
