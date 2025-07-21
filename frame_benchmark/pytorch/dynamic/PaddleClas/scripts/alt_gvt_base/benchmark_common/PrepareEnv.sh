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
pip install timm==0.4.5
# fix bug to adapt to torch 2.0
sed -i 's/local_rank/local-rank/g' train.py
dataset_url="https://paddle-imagenet-models-name.bj.bcebos.com/data/ImageNet1k/ILSVRC2012_val.tar"

# prepare data
rm -rf data
mkdir data && cd data
wget -c ${dataset_url}
tar xf ILSVRC2012_val.tar
ln -s ILSVRC2012_val imagenet
cd imagenet
ln -s ILSVRC2012_val_dirs train
ln -s ILSVRC2012_val_dirs val
cd ../../
echo "*******prepare benchmark end***********"
