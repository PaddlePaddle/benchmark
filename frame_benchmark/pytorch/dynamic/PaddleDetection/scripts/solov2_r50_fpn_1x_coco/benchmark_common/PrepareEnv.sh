#!/usr/bin/env bash
# 执行路径在模型库的根目录下
################################# 安装框架 如:
echo "*******prepare benchmark start ***********"
pip install -U pip
echo `pip --version`
pip install Cython pycocotools
# solov2模型最高只能运行在torch 1.9.0版本
pip install torch==1.9.0 torchvision==0.10.0
pip install -r requirements.txt
pip install -v -e .
pip install opencv-python --force-reinstall

################################# 准备训练数据 如:
wget -nc -P data/coco/ https://paddledet.bj.bcebos.com/data/coco_benchmark.tar
cd ./data/coco/ && tar -xf coco_benchmark.tar && mv -u coco_benchmark/* .
rm -rf coco_benchmark/ && cd ../../
echo "*******prepare benchmark end***********"
