#!/usr/bin/env bash

# install env
echo "*******prepare benchmark start ***********"
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
pip install -U pip
echo `pip --version`

wget -c --no-proxy ${FLAG_TORCH_WHL_URL}
tar_file_name=$(echo ${FLAG_TORCH_WHL_URL} | awk -F '/' '{print $NF}')
dir_name=$(echo ${tar_file_name} | awk -F '.' '{print $1}')
tar xf ${tar_file_name}
rm -rf ${tar_file_name}
pip install ${dir_name}/*

export http_proxy=${HTTP_PRO}
export https_proxy=${HTTPS_PRO}
git submodule init
git submodule update
unset http_proxy
unset https_proxy
python -m pip install -e detectron2
pip install -e .


rm -rf ./datasets/coco/
wget --no-proxy -nc -P ./datasets/coco/ https://bj.bcebos.com/v1/paddledet/data/cocomini.zip --no-check-certificate
cd ./datasets/coco/ && unzip cocomini.zip
mv -u cocomini/* ./
cd ../../

echo "*******prepare benchmark end***********"
