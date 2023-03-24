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
pip install git+https://github.com/cocodataset/panopticapi.git
unset https_proxy && unset http_proxy
pip install -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
cd -

echo "******* prepare dataset for benchmark ***********"

cd datasets
rm -rf coco
if [ ! -f "mini_coco.zip" ];then
  wget https://paddleseg.bj.bcebos.com/tipc/data/mini_coco.zip --no-check-certificate
fi
unzip -o mini_coco.zip
cd -

echo "******* prepare benchmark end *******"
