#!/usr/bin/env bash

# install env
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
echo "*******prepare benchmark start ***********"
pip install -U pip
echo `pip --version`


pip install https://paddle-wheel.bj.bcebos.com/benchmark/torch-1.9.1%2Bcu111-cp37-cp37m-linux_x86_64.whl
pip install https://paddle-wheel.bj.bcebos.com/benchmark/torchvision-0.10.1%2Bcu111-cp37-cp37m-linux_x86_64.whl


# mmcv-full wheel takes too long to compile online,
# however this wheel relies on compile environment
# different compiled wheels are provided for different cluster
if [ `nvidia-smi --list-gpus | grep A100 | wc -l` -ne "0" ]; then
    echo "Run on A100 Cluster"
    wget https://paddle-wheel.bj.bcebos.com/benchmark/mmcv_full-1.4.4-cp37-cp37m-linux_x86_64_A100.whl -O mmcv_full-1.4.4-cp37-cp37m-linux_x86_64.whl
else
    echo "Run on V100 Cluster"
    wget https://paddle-wheel.bj.bcebos.com/benchmark/mmcv_full-1.4.4-cp37-cp37m-linux_x86_64_V100.whl -O mmcv_full-1.4.4-cp37-cp37m-linux_x86_64.whl
fi
pip install mmcv_full-1.4.4-cp37-cp37m-linux_x86_64.whl && rm -f mmcv_full-1.4.4-cp37-cp37m-linux_x86_64.whl

wget https://paddle-wheel.bj.bcebos.com/benchmark/mmdetection-2.24.1.zip
unzip mmdetection-2.24.1.zip
rm -rf mmdetection-2.24.1.zip
cp mmdetection-2.24.1 mmdetection
cd mmdetection
pip install -r requirements/build.txt
python setup.py develop
cd ..

pip install mmsegmentation==0.20.2

wget https://paddle-wheel.bj.bcebos.com/benchmark/mmdetection3d-0.17.1.zip
unzip mmdetection3d-0.17.1.zip
rm -rf mmdetection3d-0.17.1.zip
cp mmdetection3d-0.17.1 mmdetection3d
cd mmdetection3d
pip install -r requirements/build.txt
python setup.py develop
cd ..

mkdir ckpts
cd ckpts
wget https://paddle-wheel.bj.bcebos.com/benchmark/fcos3d_vovnet_imgbackbone-remapped.pth
cd ..

################################# 准备训练数据 如:
mkdir -p data/
# 由于nuscenes数据集太大，为避免每次下载过于耗时，请将nuscenes数据集拷贝到data目录下
# cp -r /nuscenes_dataste_root data/
# 并软链到/data/Dataset/nuScenes目录
mkdir -p /data/Dataset
ln -s ./data/nuscenes /data/Dataset/nuScenes
echo "download data" #waiting data process
echo "dataset prepared done"
echo "*******prepare benchmark end***********"
