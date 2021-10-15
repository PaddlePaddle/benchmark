#!/usr/bin/env bash

# 公共配置文件,配置python 安装pytorch,运行目录:/workspace (起容器的时候映射的目录:benchmark/OtherFrameworks/PyTorch/)
echo "*******prepare benchmark***********"

################################# 创建一些log目录,如:
export BENCHMARK_ROOT=/workspace
log_date=`date "+%Y.%m%d.%H%M%S"`
frame=pytorch1.0.0
cuda_version=10.2
save_log_dir=${BENCHMARK_ROOT}/logs/${frame}_${log_date}_${cuda_version}/

if [[ -d ${save_log_dir} ]]; then
    rm -rf ${save_log_dir}
fi
# this for update the log_path coding mat
export TRAIN_LOG_DIR=${save_log_dir}/train_log
mkdir -p ${TRAIN_LOG_DIR}

log_path=${TRAIN_LOG_DIR}

################################# 配置python, 如:
rm -rf run_env
mkdir run_env
ln -s $(which python3.7) run_env/python
ln -s $(which pip3.7) run_env/pip
export PATH=/workspace/run_env:${PATH}

################################# 安装框架 如:

pip install -U pip
echo `pip --version`

git clone https://github.com/lzzyzlbb/stylegan2-pytorch
cd stylegan2-pytorch
pip install torch==1.3.1
pip install torchvision==0.4.2
pip install lmdb
pip install Ninja
pip install tqdm

################################# 准备训练数据 如:

mkdir -p data
wget https://paddlegan.bj.bcebos.com/datasets/stylegan_process.tar \
    -O data/ffhq.tar
tar -vxf data/ffhq.tar -C data/

echo "*******prepare benchmark end***********"

