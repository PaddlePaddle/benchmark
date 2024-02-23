#!/usr/bin/env bash

pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple

echo "*******prepare benchmark start ***********"

pip install -U pip
echo `pip --version`

pip install torch==2.0.1 torchvision==0.15.2
pip install -U openmim
mim install -r requirements/mminstall.txt
mim install -r requirements/albu.txt
mim install -v -e .

rm -rf data
wget -nc -P data https://paddledet.bj.bcebos.com/data/uapi_benchmark/benchmark_coco.tar

( cd ./data && tar -xf benchmark_coco.tar )
echo "*******prepare benchmark end***********"
