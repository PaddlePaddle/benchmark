#!/usr/bin/env bash

stage=0
stop_stage=100

# 公共配置文件,配置python 安装pytorch,运行目录:/workspace (起容器的时候映射的目录:benchmark/OtherFrameworks/PyTorch/)
echo "*******prepare benchmark***********"
################################# 创建一些log目录
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    export BENCHMARK_ROOT=/workspace   # 起容器的时候映射的目录  benchmark/OtherFrameworks/PyTorch/
    log_date=`date "+%Y.%m%d.%H%M%S"`
    frame=pytorch1.9.0
    cuda_version=10.2
    save_log_dir=${BENCHMARK_ROOT}/logs/${frame}_${log_date}_${cuda_version}/

    if [[ -d ${save_log_dir} ]]; then
        rm -rf ${save_log_dir}
    fi

    # this for update the log_path coding mat
    export TRAIN_LOG_DIR=${save_log_dir}/train_log
    mkdir -p ${TRAIN_LOG_DIR}

    log_path=${TRAIN_LOG_DIR}
fi


################################# 配置python
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    rm -rf run_env
    mkdir run_env
    ln -s $(which python3.7) run_env/python
    ln -s $(which pip3.7) run_env/pip
    # export PATH=/workspace/run_env:${PATH}
    export PATH=${BENCHMARK_ROOT}/run_env:${PATH}
fi

################################# 安装框架 如:

# 安装 pyrtorch
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    pip install -U pip
    echo `pip --version`
    pip install torch==1.9.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
fi
# 拉取模型代码并安装
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    #git submodule init
    #git submodule update
    pushd ./models/Parakeet/PWGAN/ParallelWaveGAN
    echo "--------------------------------$PWD"
    git reset --hard 8b7636b14b316ebb762c062abdb23645f6e45934
    pip install -e .
    popd

    # If you want to use distributed training, please run following command to install apex.
    git clone https://github.com/NVIDIA/apex.git
    pushd apex
    git checkout 3303b3e
    pip install -v --disable-pip-version-check --no-cache-dir ./
    pushd ../

    # install nkf
    apt-get install nkf -y
    apt-get install sox -y
    apt-get install jq -y

fi

echo "*******prepare benchmark end***********"
