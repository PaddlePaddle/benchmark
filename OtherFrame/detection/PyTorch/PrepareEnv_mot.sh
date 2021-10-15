#!/usr/bin/env bash

# 公共配置文件,配置python 安装pytorch,运行目录:/workspace (起容器的时候映射的目录:benchmark/OtherFrameworks/detection/PyTorch/)
echo "*******prepare benchmark***********"

################################# 创建一些log目录,如:
export BENCHMARK_ROOT=/workspace
log_date=`date "+%Y.%m%d.%H%M%S"`
frame=pytorch1.7.0
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
python3.7 -m pip install -U pip
echo `python3.7 -m pip --version`
pip3.7 install torch==1.7.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
pip3.7 install torchvision==0.8.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html

################################# 克隆并安装竞品
# 根据主项目的配置信息，拉取更新子模块中的代码。
git submodule init
git submodule update

################################# 准备训练数据 如:
wget -nc -P ${BENCHMARK_ROOT}/data/mot/ https://paddledet.bj.bcebos.com/data/mot_benchmark.tar

cd ${BENCHMARK_ROOT}/models/jde
mkdir -p data/mot
cp  ${BENCHMARK_ROOT}/data/mot/mot_benchmark.tar data/mot
cd ./data/mot/ && tar -xvf mot_benchmark.tar && mv -u mot_benchmark/* .
rm -rf mot_benchmark/

cd ${BENCHMARK_ROOT}/models/fairmot
mkdir -p data/mot
cp  ${BENCHMARK_ROOT}/data/mot/mot_benchmark.tar data/mot
cd ./data/mot/ && tar -xvf mot_benchmark.tar && mv -u mot_benchmark/* .
rm -rf mot_benchmark/

cd ${BENCHMARK_ROOT}

echo "*******prepare benchmark end***********"