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
# fix bug to adapt to torch 2.0
sed -i 's/local_rank/local-rank/g' train.py

# install env
# pip install -r requirements.txt
# pip install pyyaml

# # CUDA 11.1 安装高版本报cuda error
# pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
# pip install -U opencv_python==3.4.10.37
# pip list
# dataset_url="https://paddle-imagenet-models-name.bj.bcebos.com/data/ImageNet1k/ILSVRC2012_val.tar"

# prepare data
rm -rf ILSVRC2012_val.tar
rm -rf ILSVRC2012_val
rm -rf results_mobilevit_small

wget -c ${dataset_url}
tar xf ILSVRC2012_val.tar

echo "*******prepare benchmark end***********"
