#!/usr/bin/env bash

# install env
wget -c --no-proxy ${FLAG_TORCH_WHL_URL}
tar_file_name=$(echo ${FLAG_TORCH_WHL_URL} | awk -F '/' '{print $NF}')
dir_name=$(echo ${tar_file_name} | awk -F '.' '{print $1}')
tar xf ${tar_file_name}
rm -rf ${tar_file_name}

pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
pip install ${dir_name}/*
pip install -r requirements.txt

# prepare data
dataset_url="https://paddle-wheel.bj.bcebos.com/benchmark/ILSVRC2012_w.tar"
wget -c ${dataset_url} --no-proxy
tar xf ILSVRC2012_w.tar
rm -f ILSVRC2012_w.tar

echo "*******prepare benchmark end***********"
