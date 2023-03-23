#!/usr/bin/env bash

# install env
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
echo "*******prepare benchmark start ***********"
pip install -U pip
echo `pip --version`


pip install https://paddle-wheel.bj.bcebos.com/benchmark/torch-1.9.1%2Bcu111-cp37-cp37m-linux_x86_64.whl
pip install https://paddle-wheel.bj.bcebos.com/benchmark/torchvision-0.10.1%2Bcu111-cp37-cp37m-linux_x86_64.whl
pip install setuptools==59.5.0

pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.1/index.html

wget https://paddle-wheel.bj.bcebos.com/benchmark/mmdetection-2.24.1.zip
unzip mmdetection-2.24.1.zip
rm -rf mmdetection-2.24.1.zip
mv mmdetection-2.24.1 mmdetection
cd mmdetection
pip install -r requirements/build.txt
pip install terminaltables
python setup.py develop
cd ..

pip install pycocotools
pip install mmsegmentation==0.20.2

wget https://paddle-wheel.bj.bcebos.com/benchmark/mmdetection3d-0.17.1.zip
unzip mmdetection3d-0.17.1.zip
rm -rf mmdetection3d-0.17.1.zip
mv mmdetection3d-0.17.1 mmdetection3d
cd mmdetection3d
pip install -r requirements/build.txt
pip install trimesh==2.35.39
pip install tensorboard
pip install scikit-image
pip install nuscenes-devkit
pip install numpy==1.19
pip install numba==0.48.0
pip install networkx==2.2
pip install plotly
pip install pandas
pip install black
pip install flake8
pip install pytest
pip install importlib-metadata==4.2
python setup.py develop
cd ..

pip install einops
mkdir ckpts
cd ckpts
wget https://paddle-wheel.bj.bcebos.com/benchmark/fcos3d_vovnet_imgbackbone-remapped.pth
cd ..

################################# 准备训练数据 如:
mkdir -p data/
mkdir -p /data/Dataset/
# 由于nuscenes数据集太大，为避免每次下载过于耗时，请将nuscenes数据集下载后，软链到/data/Dataset/nuScenes
# ln -s /nuscenes_dataste_root /data/Dataset/nuScenes
# 并软链到data/nuscenes目录
ln -s /data/Dataset/nuScenes ./data/nuscenes
echo "download data" #waiting data process
echo "dataset prepared done"
echo "*******prepare benchmark end***********"
