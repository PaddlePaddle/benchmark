#!/usr/bin/env bash
# 执行路径在模型库的根目录下
################################# 安装框架 如:
wget -c --no-proxy ${FLAG_TORCH_WHL_URL}
tar_file_name=$(echo ${FLAG_TORCH_WHL_URL} | awk -F '/' '{print $NF}')
dir_name=$(echo ${tar_file_name} | awk -F '.' '{print $1}')
tar xf ${tar_file_name}
rm -rf ${tar_file_name}
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
pip install ${dir_name}/*

cd ./mmdetection-2.11.0
python -m pip install -v -e .
cd ..
python -m pip install -v -e .

wget  -nc https://paddleocr.bj.bcebos.com/dataset/mmcv-1.3.4-py2.py3-none-any.whl
python -m pip install mmcv-1.3.4-py2.py3-none-any.whl
# 下载数据集并解压
wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dataset/StructureLabel_val_500.tar --no-check-certificate
cd ./train_data/ && tar xf StructureLabel_val_500.tar
cd ../

export MASTER_ADDR="localhost"
export MASTER_PORT="6006"
