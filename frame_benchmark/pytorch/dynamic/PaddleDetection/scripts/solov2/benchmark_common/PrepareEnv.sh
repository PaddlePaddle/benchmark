#!/usr/bin/env bash
# 执行路径在模型库的根目录下
################################# 安装框架 如:
echo "*******prepare benchmark start ***********"
pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
echo `pip --version`
pip install torch==1.9.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torchvision==0.10.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements/build.txt
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
pip install -v -e .

################################# 准备训练数据 如:
wget -nc -P data/coco/ https://paddledet.bj.bcebos.com/data/coco_benchmark.tar
cd ./data/coco/ && tar -xf coco_benchmark.tar && mv -u coco_benchmark/* .
rm -rf coco_benchmark/ && cd ../../
echo "*******prepare benchmark end***********"
