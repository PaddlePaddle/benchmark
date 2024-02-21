#!/usr/bin/env bash

pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple

echo "*******prepare benchmark start ***********"

pip install -U pip
echo `pip --version`

pip install https://paddle-wheel.bj.bcebos.com/benchmark/torch-2.0.0.dev20230118%2Bcu117-cp37-cp37m-linux_x86_64.whl
pip install https://paddle-wheel.bj.bcebos.com/benchmark/torchvision-0.15.0.dev20230118%2Bcu117-cp37-cp37m-linux_x86_64.whl
pip install pycocotools PyYAML scipy

rm -rf data
wget -nc -P data https://paddledet.bj.bcebos.com/data/uapi_benchmark/benchmark_coco.tar

( cd ./data && tar -xf benchmark_coco.tar )
echo "*******prepare benchmark end***********"
