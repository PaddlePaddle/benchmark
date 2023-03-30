#!/usr/bin/env bash

install env
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
echo "*******prepare benchmark start ***********"
pip install -U pip
echo `pip --version`

wget -nc ${FLAG_TORCH_WHL_URL}
tar -xvf torch_dev_whls.tar
python -m pip install torch_dev_whls/*
pip install Cython

pip install openmim
pip install setuptools==50.3.2



# mmcv-full wheel takes too long to compile online,
# however this wheel relies on compile environment
# different compiled wheels are provided for different cluster
if [ `nvidia-smi --list-gpus | grep A100 | wc -l` -ne "0" ]; then
    echo "Run on A100 Cluster"
    wget https://paddle-wheel.bj.bcebos.com/benchmark/mmcv_full-1.7.1-cp37-cp37m-linux_x86_64_A100_cuda117.whl -O mmcv_full-1.7.1-cp37-cp37m-linux_x86_64.whl
    pip install mmcv_full-1.7.1-cp37-cp37m-linux_x86_64.whl && rm -f mmcv_full-1.7.1-cp37-cp37m-linux_x86_64.whl
else
    echo "Run on V100 Cluster"
    wget https://paddle-wheel.bj.bcebos.com/benchmark/mmcv_full-1.5.0-cp37-cp37m-linux_x86_64_V100.whl -O mmcv_full-1.5.0-cp37-cp37m-linux_x86_64.whl
    pip install mmcv_full-1.5.0-cp37-cp37m-linux_x86_64.whl && rm -f mmcv_full-1.5.0-cp37-cp37m-linux_x86_64.whl
fi
pip install -e .

# Download pretrained weights
mkdir -p /root/.cache/torch/hub/checkpoints/
wget https://paddle-wheel.bj.bcebos.com/benchmark/resnet50-0676ba61.pth -O /root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
# prepare data
rm -rf data
wget -nc -P data https://bj.bcebos.com/v1/paddledet/data/cocomini.zip
cd ./data && unzip cocomini.zip && mv cocomini coco && cd ..
echo "*******prepare benchmark end***********"
