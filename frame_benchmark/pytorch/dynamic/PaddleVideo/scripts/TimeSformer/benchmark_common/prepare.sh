#!/usr/bin/env bash

echo "*******prepare benchmark start ***********"
################################# 安装最新版pip
pip install -U pip
echo `pip --version`
################################# 安装torch 1.8.1
pip install torch==1.8.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html

################################# 安装TimeSformer的环境依赖
pip install -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple --no-deps --no-cache-dir

################################# 安装TimeSformer到sitepackages
python setup.py build develop

################################# 下载预训练模型到目录下
wget -nc https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth

################################# 准备训练数据
wget -nc https://videotag.bj.bcebos.com/Data/k400_videos_small.tar
tar -xf k400_videos_small.tar
echo "*******prepare benchmark end***********"