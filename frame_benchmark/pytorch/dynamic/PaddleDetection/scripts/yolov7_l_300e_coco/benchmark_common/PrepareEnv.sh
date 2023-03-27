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

wget -nc -P coco  https://paddledet.bj.bcebos.com/data/tipc/cocomini_yolov5_yolov7.tar
cd ./coco
tar -xf cocomini_yolov5_yolov7.tar 
mv cocomini_yolov5_yolov7/* ./
rm -rf *.cache
cd ..
rm -rf ./runs
echo "*******prepare benchmark end***********"
