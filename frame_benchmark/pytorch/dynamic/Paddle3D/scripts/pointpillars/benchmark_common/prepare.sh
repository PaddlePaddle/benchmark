#!/usr/bin/env bash

# install env
echo "*******prepare benchmark start ***********"
pip install -U pip
echo `pip --version`


cd ..
unset https_proxy && unset http_proxy
if [ ! -f "torch_dev_whls.tar" ];then
  wget ${FLAG_TORCH_WHL_URL}
fi
export https_proxy=${HTTP_PRO} && export http_proxy=${HTTPS_PRO}
tar -xf torch_dev_whls.tar
for whl_file in torch_dev_whls/*
do
  pip install ${whl_file}
done
pip install kornia
unset https_proxy && unset http_proxy
pip install spconv-cu117	
python setup.py develop


# 由于kitti数据集太大，为避免每次下载过于耗时，请将kitti数据集下载后，软链到/data/Dataset/kitti
mkdir -p /data/Dataset
if [ ! -d "/data/Dataset/KITTI_800" ]; then
    cd /data/Dataset
    wget https://paddle3d.bj.bcebos.com/TIPC/dataset/KITTI_800.tar --no-check-certificate
    tar -xf KITTI_800.tar
    cd -
fi
rm -rf data/KITTI_800
ln -s /data/Dataset/KITTI_800 data/KITTI_800
cd -
echo "*******prepare benchmark end***********"
