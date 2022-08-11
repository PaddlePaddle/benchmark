#!/usr/bin/env bash

# install env
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install openmim
pip install -U setuptools>=49.6.0
mim install -e .
mim install -r requirements/mminstall.txt
pip install opencv-python==4.5.5.64

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
mkdir meta && cd meta
cp ../val_list.txt val.txt
cd ../../../

echo "*******prepare benchmark end***********"
