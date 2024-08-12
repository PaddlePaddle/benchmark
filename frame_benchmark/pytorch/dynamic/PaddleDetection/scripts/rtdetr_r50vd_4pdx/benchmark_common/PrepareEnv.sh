#!/usr/bin/env bash

pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple

echo "*******prepare benchmark start ***********"

pip install -U pip
echo `pip --version`

pip install torch==2.0.1 torchvision==0.15.2  # 升级torchvision 会报错cannot import name 'datapoints' from 'torchvision'
pip install pycocotools==2.0.8 PyYAML==6.0 scipy==1.14.0

rm -rf data
wget -nc -q -P data https://paddledet.bj.bcebos.com/data/uapi_benchmark/benchmark_coco.tar
wget  https://paddle-qa.bj.bcebos.com/benchmark/pretrained/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth
cp ResNet50_vd_ssld_v2_pretrained_from_paddle.pth /root/.cache/torch/hub/checkpoints/

( cd ./data && tar -xf benchmark_coco.tar )
echo "*******prepare benchmark end***********"
