#!/usr/bin/env bash

unset http_proxy https_proxy
dataset_url="https://paddle-imagenet-models-name.bj.bcebos.com/data/ImageNet1k/ILSVRC2012_val.tar"

# prepare data
rm -rf data
mkdir data && cd data

if [ ${RUN_PLAT} == "local" ]; then
    ln -s ${all_path}/dataset/otherframe/ILSVRC2012_val ./
else
    wget -c ${dataset_url}
    tar xf ILSVRC2012_val.tar
fi

ln -s ILSVRC2012_val imagenet
cd imagenet
ln -s ILSVRC2012_val_dirs train
ln -s ILSVRC2012_val_dirs val
cd ../../
