#!/usr/bin/env bash

echo "******* install enviroments for benchmark ***********"
echo `pip --version`

unset https_proxy && unset http_proxy
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U openmim
mim install -v mmcv-full==1.5.0
pip install timm==0.4.12
pip install mmdet==2.22.0
pip install mmsegmentation==0.20.2

ln -s ../detection/ops ./
cd ops && sh make.sh && cd -
cp dist_train.sh dist_train.sh

echo "******* prepare dataset for benchmark ***********"

rm -rf data
mkdir -p data/ade
cd data/ade
wget https://paddleseg.bj.bcebos.com/dataset/ADEChallengeData2016.zip --no-check-certificate
unzip -o ADEChallengeData2016.zip
cd -

echo "******* prepare benchmark end *******"
