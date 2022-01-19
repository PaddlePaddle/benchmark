#!/usr/bin/env bash

# install env
pip install mxnet-cu102==1.8.0
pip install gluoncv --upgrade
sudo apt-get install -y libopenblas-dev
# 需注意的是，mxnet官方没有提供最新对docker镜像，只能使用现有的paddle镜像进行安装，注意cudnn是否安装正确,可能需要手动安装
