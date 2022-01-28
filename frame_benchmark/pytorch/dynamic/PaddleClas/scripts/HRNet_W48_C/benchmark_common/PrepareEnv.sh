#!/usr/bin/env bash

# install env
pip install torch torchvision
sed -i "s/opencv-python==3.4.1.15/opencv-python==4.4.0.46/g" requirements.txt
sed -i "s/shapely==1.6.4/shapely/g" requirements.txt
pip install -r requirements.txt


dataset_url="https://paddle-imagenet-models-name.bj.bcebos.com/data/ImageNet1k/ILSVRC2012_val.tar"
# prepare data
rm -rf imagenet
mkdir imagenet && cd imagenet
wget -c ${dataset_url}
tar xf ILSVRC2012_val.tar
ln -s ILSVRC2012_val images
cd images
ln -s ILSVRC2012_val_dirs train
ln -s ILSVRC2012_val_dirs val
cd ../../
