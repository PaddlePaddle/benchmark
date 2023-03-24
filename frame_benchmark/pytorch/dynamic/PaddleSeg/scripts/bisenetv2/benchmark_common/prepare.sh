#!/usr/bin/env bash

echo "******* install enviroments for benchmark ***********"
echo `pip --version`

if [ ! -f "torch_dev_whls.tar" ];then
  unset https_proxy && unset http_proxy
  wget ${FLAG_TORCH_WHL_URL}
fi
tar -xf torch_dev_whls.tar
export https_proxy=${PROXY_IP} && export http_proxy=${PROXY_IP}
for whl_file in torch_dev_whls/*
do
  pip install ${whl_file}
done
unset https_proxy && unset http_proxy
pip install ninja -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -v mmcv-full==1.7.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -v -e .

echo "******* prepare dataset for benchmark ***********"
if [ ! -f "mmseg_benchmark_configs.zip" ];then
  wget https://paddleseg.bj.bcebos.com/benchmark/mmseg/mmseg_benchmark_configs.zip
fi
unzip -o mmseg_benchmark_configs.zip 
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
