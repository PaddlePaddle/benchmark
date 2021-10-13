#!/usr/bin/env bash

# prepare data
dataset_url=$1
rm -rf data
mkdir data && cd data
wget -c ${dataset_url}
tar xf ILSVRC2012_val.tar
ln -s ILSVRC2012_val/ILSVRC2012_val_dirs train
ln -s ILSVRC2012_val/ILSVRC2012_val_dirs val
cd ../
