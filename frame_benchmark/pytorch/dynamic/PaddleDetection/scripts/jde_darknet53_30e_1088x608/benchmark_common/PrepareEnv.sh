#!/usr/bin/env bash
# 执行路径在模型库的根目录下
################################# 安装框架 如:
echo "*******prepare benchmark start ***********"
#pip install -U pip
echo `pip --version`
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade setuptools
pip install Cython pycocotools matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
#pip install torch==1.10.0 torchvision==0.11.1
wget https://paddle-wheel.bj.bcebos.com/benchmark/torch-1.10.0%2Bcu111-cp37-cp37m-linux_x86_64.whl
wget https://paddle-wheel.bj.bcebos.com/benchmark/torchvision-0.11.1%2Bcu111-cp37-cp37m-linux_x86_64.whl
pip install torch-1.10.0+cu111-cp37-cp37m-linux_x86_64.whl
pip install torchvision-0.11.1+cu111-cp37-cp37m-linux_x86_64.whl
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple some-package

################################# 准备训练数据 如:
wget -nc -P dataset/mot https://paddledet.bj.bcebos.com/data/mot_benchmark.tar
cd ./dataset/mot && tar -xf mot_benchmark.tar && mv -u mot_benchmark/* .
rm -rf mot_benchmark/ && cd ../../
################################# 准备预训练模型 如:
wget -nc -P weights/ https://paddledet.bj.bcebos.com/models/pretrained/darknet53.conv.74
echo "*******prepare benchmark end***********"
