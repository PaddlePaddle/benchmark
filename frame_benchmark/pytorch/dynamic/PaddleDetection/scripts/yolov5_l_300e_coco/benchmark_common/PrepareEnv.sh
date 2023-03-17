#!/usr/bin/env bash

# install env
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
echo "*******prepare benchmark start ***********"
pip install -U pip
echo `pip --version`

wget -nc ${FLAG_TORCH_WHL_URL}
tar -xvf torch_dev_whls.tar
python -m pip install torch_dev_whls/*
rm -rf torch_dev_whls*
pip install -r requirements.txt

wget -nc -P datasets  https://bj.bcebos.com/v1/paddledet/data/coco128_yolov5_yolov7.tar
cd ./datasets && tar -xf coco128_yolov5_yolov7.tar 
cd coco128 && mkdir images && mv train2017 images/ && mv val2017 images/
cd .. && cd ..
rm -rf ./datasets/coco128/labels/*.cache
rm -rf ./runs
echo "*******prepare benchmark end***********"
