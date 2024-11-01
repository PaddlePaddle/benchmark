#!/usr/bin/env bash

pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple

echo "*******prepare benchmark start ***********"
pip install -U pip
echo `pip --version`

# install env
wget -c --no-proxy ${FLAG_TORCH_WHL_URL}
tar_file_name=$(echo ${FLAG_TORCH_WHL_URL} | awk -F '/' '{print $NF}')
dir_name=$(echo ${tar_file_name} | awk -F '.' '{print $1}')
tar xf ${tar_file_name}
rm -rf ${tar_file_name}

pip install ${dir_name}/*
pip install -e .
rm -rf /root/.config/Ultralytics
wget --no-proxy https://paddledet.bj.bcebos.com/data/tipc/Arial.ttf
mkdir -p /root/.config/Ultralytics/
mv Arial.ttf /root/.config/Ultralytics/


rm -rf data
wget -nc -q -P data https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/benchmark/dataset/coco_train_benchmark.tar

( cd ./data && tar -xf coco_train_benchmark.tar )
echo "*******prepare benchmark end***********"
