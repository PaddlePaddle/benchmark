#!/usr/bin/env bash

# install env
pip install -r requirements.txt

dataset_url="https://paddle-imagenet-models-name.bj.bcebos.com/data/ImageNet1k/ILSVRC2012_val.tar"

# prepare data
rm -rf ILSVRC2012_val.tar
rm -rf ILSVRC2012_val
rm -rf results_mobilevit_small

wget -c ${dataset_url}
tar xf ILSVRC2012_val.tar

echo "*******prepare benchmark end***********"
