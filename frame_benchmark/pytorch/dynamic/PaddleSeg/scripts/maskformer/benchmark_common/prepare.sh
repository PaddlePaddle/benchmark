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

export https_proxy=${PROXY_IP} && export http_proxy=${PROXY_IP}
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
unset https_proxy && unset http_proxy
pip install -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "******* prepare dataset for benchmark ***********"

cd datasets
rm -rf ADE20k-20 ADEChallengeData2016
if [ ! -f "ADE20k-20.zip" ];then
  wget https://bj.bcebos.com/paddleseg/tipc/data/ADE20k-20.zip --no-check-certificate
fi
unzip -o ADE20k-20.zip 
mv ADE20k-20 ADEChallengeData2016 
cd -
python datasets/prepare_ade20k_sem_seg.py

echo "******* prepare benchmark end *******"
