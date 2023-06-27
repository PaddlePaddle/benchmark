#!/usr/bin/env bash

# install env
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
echo "*******prepare benchmark start ***********"
pip install -U pip
echo `pip --version`

wget -c --no-proxy ${FLAG_TORCH_WHL_URL}
tar_file_name=$(echo ${FLAG_TORCH_WHL_URL} | awk -F '/' '{print $NF}')
dir_name=$(echo ${tar_file_name} | awk -F '.' '{print $1}')
tar xf ${tar_file_name}
rm -rf ${tar_file_name}
pip install ${dir_name}/*

pip install -r requirements.txt

rm -rf datasets
wget -nc -P datasets/coco128/  https://paddledet.bj.bcebos.com/data/tipc/cocomini_yolov5_yolov7.tar
cd ./datasets/coco128/ && tar -xf cocomini_yolov5_yolov7.tar
mv cocomini_yolov5_yolov7/* ./
cd .. && cd ..
rm -rf ./datasets/coco128/labels/*.cache
rm -rf ./runs
wget https://paddledet.bj.bcebos.com/data/tipc/Arial.ttf
mkdir -p /root/.config/Ultralytics/
mv Arial.ttf /root/.config/Ultralytics/

# fix bug to adapt to torch 2.0
sed -i 's/local_rank/local-rank/g' train.py

echo "*******prepare benchmark end***********"
