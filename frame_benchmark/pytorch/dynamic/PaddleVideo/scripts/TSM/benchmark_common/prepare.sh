#!/usr/bin/env bash
# 执行路径在模型库的根目录下
################################# 安装框架 如:
echo "*******prepare benchmark start ***********"
pip install -U pip
echo `pip --version`

# pip install torch==1.9.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install https://paddle-wheel.bj.bcebos.com/benchmark/torch-1.12.0%2Bcu113-cp37-cp37m-linux_x86_64.whl
pip install https://paddle-wheel.bj.bcebos.com/benchmark/torchvision-0.13.0%2Bcu113-cp37-cp37m-linux_x86_64.whl

pip install -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple

################################# 准备训练数据 如:
wget https://videotag.bj.bcebos.com/Data/k400_rawframes_small.tar \
    -O data/k400_rawframes_small.tar
tar -zxvf data/k400_rawframes_small.tar -C data/
echo "*******prepare benchmark end***********"