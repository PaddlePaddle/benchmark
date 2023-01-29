#!/usr/bin/env bash

# install env
pip install -r requirements.txt
pip install pyyaml

# CUDA 11.1 安装高版本报cuda error
pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
pip install -U opencv_python==3.4.10.37
pip list
dataset_url="https://paddle-imagenet-models-name.bj.bcebos.com/data/ImageNet1k/ILSVRC2012_val.tar"

# prepare data
rm -rf ILSVRC2012_val.tar
rm -rf ILSVRC2012_val
rm -rf results_mobilevit_small

wget -c ${dataset_url}
tar xf ILSVRC2012_val.tar

echo "*******prepare benchmark end***********"
