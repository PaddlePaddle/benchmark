#!/usr/bin/env bash
# 执行路径在模型库的根目录下
################################# 安装框架 如:
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

pip install -r requirements.txt
pip install -v -e .

################################# 准备训练数据 如:
wget -nc -P data/coco/ https://paddledet.bj.bcebos.com/data/coco_benchmark.tar
cd ./data/coco/ && tar -xf coco_benchmark.tar && mv -u coco_benchmark/* .
rm -rf coco_benchmark/ && cd ../../
echo "*******prepare benchmark end***********"
