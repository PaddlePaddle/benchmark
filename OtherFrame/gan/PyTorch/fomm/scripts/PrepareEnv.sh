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
export https_proxy=http://172.19.57.45:3128 && export http_proxy=http://172.19.57.45:3128

pip install -U pip
echo `pip --version`

git clone https://github.com/lzzyzlbb/first-order-model
cd first-order-model
pip install -r requirements.txt

################################# 准备训练数据 如:
unset https_proxy
unset http_proxy

mkdir -p data
wget https://paddlegan.bj.bcebos.com/datasets/fom_test_data.tar \
    -O data/fom_test_data.tar
tar -vxf data/fom_test_data.tar -C data/
mv data/vox_256/ data/vox-png/

echo "*******prepare benchmark end***********"

