#!/usr/bin/env bash

################################# 安装框架 如:
pip install -U pip
echo `pip --version`
pip install https://paddle-wheel.bj.bcebos.com/benchmark/torch-1.12.0%2Bcu113-cp37-cp37m-linux_x86_64.whl
pip install https://paddle-wheel.bj.bcebos.com/benchmark/torchvision-0.13.0%2Bcu113-cp37-cp37m-linux_x86_64.whl
pip install lmdb               -i https://mirrors.ustc.edu.cn/pypi/web/simple
pip install Ninja              -i https://mirrors.ustc.edu.cn/pypi/web/simple
pip install tqdm               -i https://mirrors.ustc.edu.cn/pypi/web/simple
pip list 
################################# 准备训练数据 如:
mkdir -p data
wget https://paddlegan.bj.bcebos.com/datasets/stylegan_process.tar \
    -O data/ffhq.tar
tar -xf data/ffhq.tar -C data/
echo "*******prepare benchmark end***********"
