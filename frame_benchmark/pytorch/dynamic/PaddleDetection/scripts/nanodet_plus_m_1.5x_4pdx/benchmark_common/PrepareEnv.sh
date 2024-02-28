#!/usr/bin/env bash

pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple

echo "*******prepare benchmark start ***********"

pip install -U pip
echo `pip --version`

pip install torch==1.13.1 torchvision==0.14.1
pip install -r requirements.txt
pip install -e .

rm -rf data
wget -nc -P data https://paddledet.bj.bcebos.com/data/uapi_benchmark/benchmark_coco.tar

( cd ./data && tar -xf benchmark_coco.tar )
echo "*******prepare benchmark end***********"
