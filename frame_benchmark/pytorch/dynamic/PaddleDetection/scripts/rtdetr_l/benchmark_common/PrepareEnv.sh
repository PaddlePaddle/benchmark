#!/usr/bin/env bash

pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple

echo "*******prepare benchmark start ***********"

pip install -U pip
echo `pip --version`

pip install -e .


rm -rf data
wget -nc -q -P data https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/benchmark/dataset/coco_train_benchmark.tar

( cd ./data && tar -xf coco_train_benchmark.tar )
echo "*******prepare benchmark end***********"
