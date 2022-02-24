#!/usr/bin/env bash

# install env
pip install torch torchvision
pip install timm==0.4.5

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
