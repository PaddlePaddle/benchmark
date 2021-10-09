#!/bin/bash

function usage () {
  cat <<EOF
  usage: $0 [options]
  -h         optional   Print this help message
  -d  dynamic_models
  -s  static_models
  -c  cuda_version 9.0|10.0
  -n  dynamic_image_name
  -t  static_image_name
  -l  log_dir
  -p  all_path contains dir of prepare(pretrained models), dataset, logs
EOF
}
while getopts h:d:s:c:n:t:l:p: opt
do
  case $opt in
  h) usage; exit 0 ;;
  d) dynamic_models="$OPTARG" ;;
  s) static_models="$OPTARG" ;;
  c) cuda_version="$OPTARG" ;;
  n) dynamic_image_name="$OPTARG" ;;
  t) static_image_name="$OPTARG" ;;
  l) log_dir="$OPTARG" ;;
  p) all_path="$OPTARG" ;;
  \?) usage; exit 1 ;;
  esac
done

function prepare(){
    echo "*******prepare benchmark***********"
    rm /etc/apt/sources.list
    cp ${all_path}/sources.list /etc/apt
    apt-get update
    apt-get install libmysqlclient-dev git curl psmisc -y

    save_log_dir=${log_dir}

    if [[ -d ${save_log_dir} ]]; then
        rm -rf ${save_log_dir}
    fi

    # this for update the log_path coding mat
    export TRAIN_LOG_DIR=${save_log_dir}/train_log
    export PROFILER_LOG_DIR=${save_log_dir}/profiler_log
    
    mkdir -p ${TRAIN_LOG_DIR}
    mkdir -p ${PROFILER_LOG_DIR}

    train_log_dir=${save_log_dir}/train_log

    export ROOT_PATH=/home/pr_run
    export BENCHMARK_ROOT=${ROOT_PATH}/benchmark

    data_path=${all_path}/dataset
    prepare_path=${all_path}/prepare

    # 每个任务每个模式均做创建处理，并删除上一次任务的残存文件，避免相同repo不通分支引入的bug
    mkdir -p ${ROOT_PATH}
    cd ${ROOT_PATH}
    rm -rf benchmark
    git clone https://github.com/PaddlePaddle/benchmark.git --recursive

    cd ${BENCHMARK_ROOT}
    benchmark_commit_id=$(git log|head -n1|awk '{print $2}')
    echo "benchmark_commit_id is: "${benchmark_commit_id}

    rm -rf run_env
    mkdir run_env
    ln -s $(which python3.7) run_env/python
    ln -s $(which pip3.7) run_env/pip
    export PATH=$(pwd)/run_env:${PATH}

    pip install -U pip
    echo `pip --version`
    pip install ${all_path}/benchmark_ce/70725f72756e/thirdparty/opencv_python-4.5.2.52-cp37-cp37m-manylinux2014_x86_64.whl
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda$(echo ${cuda_version}|cut -d "." -f1)0
    # fix ssl temporarily
    if [ ${cuda_version} == 10.1 ]; then
        export LD_LIBRARY_PATH=${all_path}/tools/ssl/lib:${LD_LIBRARY_PATH}
    fi

    if python -c "import paddle" >/dev/null 2>&1
    then
        echo "paddle import success!"
    fi
    
    echo "*******prepare end!***********"
}

function run(){
    export IMPLEMENT_TYPE=static_graph
    export implement_type=static_graph
    pip uninstall paddlepaddle-gpu -y
    pip install "${static_image_name}"
    log_path=${save_log_dir}/static_graph/index
    mkdir -p ${log_path}
    source ${BENCHMARK_ROOT}/scripts/static_graph_models.sh

    # static_models , 分割的字符串切分成为数组，然后遍历执行即可。
    static_model_array=("${static_models//,/ }")
    echo "static_model_array: ${static_model_array[*]}"
    for model in ${static_model_array[*]}
    do
        echo "=====================${model} run begin=================="
        $model
        sleep 60
        echo "*********************${model} run end!!******************"
    done

    export IMPLEMENT_TYPE=dynamic_graph
    export implement_type=dynamic_graph
    pip uninstall paddlepaddle-gpu -y
    pip install "${dynamic_image_name}"
    log_path=${save_log_dir}/dynamic_graph/index
    mkdir -p ${log_path}
    source ${BENCHMARK_ROOT}/scripts/dynamic_graph_models.sh

    # dynamic_models , 分割的字符串切分成为数组，然后遍历执行即可。
    dynamic_model_array=("${dynamic_models//,/ }")
    echo "dynamic_model_array: ${dynamic_model_array[*]}"
    for model in ${dynamic_model_array[*]}
    do
        echo "=====================${model} run begin=================="
        $model
        sleep 60
        echo "*********************${model} run end!!******************"
    done
}

prepare
run
