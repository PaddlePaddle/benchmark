#!/usr/bin/env bash

################################# 安装框架 如:
pip install -U pip
echo `pip --version`
pip install -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple
imageio_download_bin ffmpeg
pip list 
################################# 准备训练数据 如:

mkdir -p data
wget https://paddlegan.bj.bcebos.com/datasets/fom_test_data.tar \
    -O data/fom_test_data.tar
tar -vxf data/fom_test_data.tar -C data/
mv data/first_order/Voxceleb/ data/vox-png/
echo "*******prepare benchmark end***********"


