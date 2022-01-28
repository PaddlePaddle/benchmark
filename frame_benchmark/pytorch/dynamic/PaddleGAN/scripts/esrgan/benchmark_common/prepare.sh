#!/usr/bin/env bash

################################# 安装框架 如:
pip install -U pip
echo `pip --version`
pip install torch==1.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchvision==0.10.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html

pip install mmcv-full==1.3.13 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html

pip install -r requirements.txt
pip install -v -e .

################################# 准备训练数据 如:
mkdir -p data
wget https://paddlegan.bj.bcebos.com/datasets/DIV2KandSet14.tar \
    -O data/DIV2KandSet14.tar
tar -vxf data/DIV2KandSet14.tar -C data/
echo "download data" #waiting data process
echo "dataset prepared done" 

echo "*******prepare benchmark end***********"


