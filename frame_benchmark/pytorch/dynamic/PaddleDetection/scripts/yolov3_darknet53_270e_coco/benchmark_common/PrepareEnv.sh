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

pip install Cython
if [ `nvidia-smi --list-gpus | grep A100 | wc -l` -ne "0" ]; then
    echo "Run on A100 Cluster"
    wget https://paddle-wheel.bj.bcebos.com/benchmark/mmcv_full-1.7.1-cp37-cp37m-linux_x86_64_A100_cuda117.whl -O mmcv_full-1.7.1-cp37-cp37m-linux_x86_64.whl
    pip install mmcv_full-1.7.1-cp37-cp37m-linux_x86_64.whl && rm -f mmcv_full-1.7.1-cp37-cp37m-linux_x86_64.whl
else
    echo "Run on V100 Cluster"
    wget https://paddle-wheel.bj.bcebos.com/benchmark/mmcv_full-1.5.0-cp37-cp37m-linux_x86_64_V100.whl -O mmcv_full-1.5.0-cp37-cp37m-linux_x86_64.whl
    pip install mmcv_full-1.5.0-cp37-cp37m-linux_x86_64.whl && rm -f mmcv_full-1.5.0-cp37-cp37m-linux_x86_64.whl
fi

# pip install openmim
pip install setuptools==50.3.2
pip install -e .

# Download pretrained weights
mkdir -p /root/.cache/torch/hub/checkpoints/
wget https://download.openmmlab.com/pretrain/third_party/darknet53-a628ea1b.pth -O /root/.cache/torch/hub/checkpoints/darknet53-a628ea1b.pth
# prepare data
rm -rf data
wget -nc -P data https://bj.bcebos.com/v1/paddledet/data/coco.tar
cd ./data && tar -xf coco.tar && cd ..
echo "*******prepare benchmark end***********"
