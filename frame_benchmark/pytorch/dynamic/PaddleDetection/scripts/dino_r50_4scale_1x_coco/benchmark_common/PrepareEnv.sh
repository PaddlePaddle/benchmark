#!/usr/bin/env bash

# install env
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
echo "*******prepare benchmark start ***********"
pip install -U pip
echo `pip --version`

pip install https://paddle-wheel.bj.bcebos.com/benchmark/torch-1.12.0%2Bcu113-cp37-cp37m-linux_x86_64.whl
pip install https://paddle-wheel.bj.bcebos.com/benchmark/torchvision-0.13.0%2Bcu113-cp37-cp37m-linux_x86_64.whl
git submodule init
git submodule update
python -m pip install -e detectron2
pip install -e .


wget -nc -P ./datasets/coco/ https://paddledet.bj.bcebos.com/data/coco_benchmark.tar --no-check-certificate
cd ./datasets/coco/ && tar -xf coco_benchmark.tar
mv -u coco_benchmark/* ./
cd ../../

echo "*******prepare benchmark end***********"
