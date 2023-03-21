#!/usr/bin/env bash

echo "******* install enviroments for benchmark ***********"
echo `pip --version`

if [ ! -f "torch_dev_whls.tar" ];then
  wget ${FLAG_TORCH_WHL_URL}
fi
tar -xf torch_dev_whls.tar
pip install torch_dev_whls/*

git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
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
