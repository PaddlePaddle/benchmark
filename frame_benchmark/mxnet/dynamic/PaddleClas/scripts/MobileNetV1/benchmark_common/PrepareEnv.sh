#!/usr/bin/env bash

# install env
pip install -U --pre mxnet -f https://dist.mxnet.io/python/cu112
pip install gluoncv --upgrade

# prepare data
dataset_url="https://paddle-imagenet-models-name.bj.bcebos.com/data/ImageNet1k/ILSVRC2012_val.tar"
rm -rf data
mkdir data && cd data
wget -c ${dataset_url}
tar xf ILSVRC2012_val.tar
ln -s ILSVRC2012_val/ILSVRC2012_val_dirs train
ln -s ILSVRC2012_val/ILSVRC2012_val_dirs val
cd ../
echo "*******prepare benchmark end***********"
