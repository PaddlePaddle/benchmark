#!/usr/bin/env bash

# install env
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
echo "*******prepare benchmark start ***********"
pip install -U pip
echo `pip --version`

pip install Cython
pip install https://paddle-wheel.bj.bcebos.com/benchmark/torch-1.12.0%2Bcu113-cp37-cp37m-linux_x86_64.whl
pip install https://paddle-wheel.bj.bcebos.com/benchmark/torchvision-0.13.0%2Bcu113-cp37-cp37m-linux_x86_64.whl
# pip install openmim
pip install setuptools==50.3.2
pip install -e .


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


# Download pretrained weights
mkdir -p /root/.cache/torch/hub/checkpoints/
wget https://paddle-wheel.bj.bcebos.com/benchmark/resnet50-0676ba61.pth -O /root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
# prepare data
rm -rf data
wget -nc -P data https://bj.bcebos.com/v1/paddledet/data/coco.tar
cd ./data && tar -xf coco.tar && cd ..
echo "*******prepare benchmark end***********"
