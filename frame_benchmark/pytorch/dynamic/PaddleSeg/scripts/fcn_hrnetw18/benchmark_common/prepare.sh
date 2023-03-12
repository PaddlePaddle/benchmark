#!/usr/bin/env bash

echo "******* install enviroments for benchmark ***********"
echo `pip --version`

if [ ! -f "torch_dev_whls.tar" ];then
  wget https://paddle-wheel.bj.bcebos.com/benchmark/torch_dev_whls.tar  
fi
tar -xf torch_dev_whls.tar
pip install torch_dev_whls/*
pip install mmcv-full==1.7.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e .

echo "******* prepare dataset for benchmark ***********"
wget https://paddleseg.bj.bcebos.com/benchmark/mmseg/mmseg_benchmark_configs.zip
unzip mmseg_benchmark_configs.zip 
cp dist_train.sh tools/dist_train.sh

if [ $(ls -lR data/cityscapes | grep "^-" | wc -l) -ne 600 ];then
  rm -rf data
  mkdir -p data
  wget https://dataset.bj.bcebos.com/benchmark/cityscapes_300imgs.tar.gz -O "data/cityscapes_300imgs.tar.gz"
  tar -zxf data/cityscapes_300imgs.tar.gz -C data/
  rm -rf data/cityscapes_300imgs.tar.gz
  mv data/cityscapes_300imgs data/cityscapes
else
  echo "******* cityscapes dataset already exists *******"
fi

echo "******* prepare benchmark end *******"
