#!/usr/bin/env bash

# 公共配置文件,配置python 安装pytorch,运行目录:/workspace (起容器的时候映射的目录:benchmark/OtherFrameworks/PyTorch/)
echo "*******prepare benchmark***********"
################################# 创建一些log目录,如:
export BENCHMARK_ROOT=/workspace   # 起容器的时候映射的目录  benchmark/OtherFrameworks/PyTorch/
log_date=`date "+%Y.%m%d.%H%M%S"`
frame=pytorch1.10.0
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
pip install torch==1.10.0 -f https://download.pytorch.org/whl/torch_stable.html

##################################  安装其他的公共依赖（单个模型的依赖在中设置,All_PyTorch_Models.sh 中）,如:
pip install git+https://github.com/huggingface/transformers -i https://mirror.baidu.com/pypi/simple
pip install accelerate -i https://mirror.baidu.com/pypi/simple
pip install datasets >= 1.8.0 -i https://mirror.baidu.com/pypi/simple
pip install sentencepiece != 0.1.92 -i https://mirror.baidu.com/pypi/simple
pip install scipy -i https://mirror.baidu.com/pypi/simple
pip install scikit-learn -i https://mirror.baidu.com/pypi/simple
pip install protobuf -i https://mirror.baidu.com/pypi/simple
echo "*******prepare benchmark end***********"
