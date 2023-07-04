#!/usr/bin/env bash

# install env
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
echo "*******prepare benchmark start ***********"
pip install -U pip
echo `pip --version`


pip install https://paddle-wheel.bj.bcebos.com/benchmark/torch-1.9.1%2Bcu111-cp37-cp37m-linux_x86_64.whl
pip install https://paddle-wheel.bj.bcebos.com/benchmark/torchvision-0.10.1%2Bcu111-cp37-cp37m-linux_x86_64.whl
pip install setuptools==59.5.0

pip install https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/mmcv_full-1.4.0-cp37-cp37m-manylinux1_x86_64.whl

wget https://paddle-wheel.bj.bcebos.com/benchmark/mmdetection-2.24.1.zip
unzip mmdetection-2.24.1.zip
rm -rf mmdetection-2.24.1.zip
mv mmdetection-2.24.1 mmdetection
cd mmdetection
export https_proxy=${HTTP_PRO}
export http_proxy=${HTTP_PRO}
pip install -r requirements/build.txt
pip install terminaltables
python setup.py develop
unset http_proxy
unset https_proxy
cd ..

pip install pycocotools
pip install mmsegmentation==0.20.2

wget https://paddle-wheel.bj.bcebos.com/benchmark/mmdetection3d-0.17.1.zip
unzip mmdetection3d-0.17.1.zip
rm -rf mmdetection3d-0.17.1.zip
mv mmdetection3d-0.17.1 mmdetection3d
cd mmdetection3d
export https_proxy=${HTTP_PRO}
export http_proxy=${HTTP_PRO}
pip install -r requirements/build.txt
pip install trimesh==2.35.39
pip install tensorboard
pip install scikit-image
pip install nuscenes-devkit
# numpy 版本升级,否则报错:ValueError: numpy.ndarray size changed, may indicate binary incompatibility
pip install --upgrade numpy
pip install numba==0.48.0
pip install networkx==2.2
pip install plotly
pip install pandas
pip install black
pip install flake8
pip install pytest
pip install importlib-metadata==4.2
python setup.py develop
unset http_proxy
unset https_proxy
cd ..

pip install einops
mkdir ckpts
cd ckpts
wget https://paddle-wheel.bj.bcebos.com/benchmark/fcos3d_vovnet_imgbackbone-remapped.pth
cd ..

################################# 准备训练数据 如:
mkdir -p data/
mkdir -p /data/Dataset/
mkdir -p /workspace/datset/nuScenes/
# 由于nuscenes数据集太大，为避免每次下载过于耗时，请将nuscenes数据集下载后，软链到/data/Dataset/nuScenes
# 并软链到data/nuscenes目录
current_path=`pwd`
cp ${BENCHMARK_ROOT}/models_data_cfs/model_benchmark/paddle3d/petr_data/nuscenes.tar ./data
cd data && tar -xvf nuscenes.tar && cd ../
ln -s ${current_path}/data/nuscenes /data/Dataset/nuScenes
ln -s ${current_path}/data/nuscenes /workspace/datset/nuScenes/nuscenes
echo "download data" #waiting data process
echo "dataset prepared done"
echo "*******prepare benchmark end***********"
