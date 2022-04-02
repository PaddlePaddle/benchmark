#!/usr/bin/env bash

# install env
pip install -r classification/requirements.txt

dataset_url="https://paddle-imagenet-models-name.bj.bcebos.com/data/ImageNet1k/ILSVRC2012_val.tar"

# prepare data
rm -rf data
mkdir data && cd data
wget -c ${dataset_url}
tar xf ILSVRC2012_val.tar
cd ILSVRC2012_val
ln -s ILSVRC2012_val_dirs train
ln -s ILSVRC2012_val_dirs val

echo "*******prepare benchmark end***********"
