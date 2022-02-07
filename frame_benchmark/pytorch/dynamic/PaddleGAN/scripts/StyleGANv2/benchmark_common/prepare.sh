#!/usr/bin/env bash

################################# 安装框架 如:
pip install -U pip
echo `pip --version`
pip install torch==1.3.1       -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torchvision==0.4.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install lmdb               -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install Ninja              -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tqdm               -i https://pypi.tuna.tsinghua.edu.cn/simple

################################# 准备训练数据 如:
mkdir -p data
wget https://paddlegan.bj.bcebos.com/datasets/stylegan_process.tar \
    -O data/ffhq.tar
tar -vxf data/ffhq.tar -C data/
echo "*******prepare benchmark end***********"


