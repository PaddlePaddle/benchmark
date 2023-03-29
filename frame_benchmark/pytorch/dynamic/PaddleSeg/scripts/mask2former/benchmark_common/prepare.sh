#!/usr/bin/env bash

echo "******* install enviroments for benchmark ***********"
echo `pip --version`


unset https_proxy && unset http_proxy
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple

export https_proxy=${HTTP_PRO} && export http_proxy=${HTTPS_PRO}
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
