#!/bin/bash

function usage () {
  cat <<EOF
  usage: $0 [options]
  -h         optional   Print this help message
  -m  model  run models
  -c  cuda_version 9.0|10.0
  -n  image_name
  -i  image_commit_id
  -a  image_branch develop|1.6|pr_number|v1.6.0
  -v  paddle_version
  -p  all_path contains dir of prepare(pretrained models), dataset, logs, such as /ssd1/ljh
  -t  job_type  benchmark_daliy | models test | pr_test
  -g  device_type  p40 | v100
  -s  implement_type of model static_graph | dynamic_graph
EOF
}
if [ $# -lt 18 ] ; then
  usage
  exit 1;
fi
while getopts h:m:c:n:i:a:v:p:t:g:s: opt
do
  case $opt in
  h) usage; exit 0 ;;
  m) model="$OPTARG" ;;
  c) cuda_version="$OPTARG" ;;
  n) image_name="$OPTARG" ;;
  i) image_commit_id="$OPTARG" ;;
  a) image_branch="$OPTARG" ;;
  v) paddle_version="$OPTARG" ;;
  p) all_path="$OPTARG" ;;
  t) job_type="$OPTARG" ;;
  g) device_type="$OPTARG" ;;
  s) implement_type="$OPTARG" ;;
  \?) usage; exit 1 ;;
  esac
done

origin_path=$(pwd)

function prepare(){
    echo "*******prepare benchmark***********"

    # this is for image paddlepaddle/paddle_manylinux_devel:cuda${cuda_version}_cudnn${cudnn_version}
    # export LD_LIBRARY_PATH=/opt/_internal/cpython-2.7.11-ucs4/lib:${LD_LIBRARY_PATH#/opt/_internal/cpython-2.7.11-ucs2/lib:}
    # export PATH=/opt/python/cp27-cp27mu/bin/:${PATH}
    # yum install mysql-devel -y
    # pip install MySQL-python
    
    
    # this is for image paddlepaddle/paddle:latest-gpu-cuda${cuda_version}-cudnn${cudnn_version}
    if [ '10.0' = ${cuda_version} -o "p40" = ${device_type} ] ; then
        export LD_LIBRARY_PATH=/home/work/418.39/lib64/:$LD_LIBRARY_PATH
    fi

    # NOTE: this path is for profiler
    # export LD_LIBRARY_PATH=/home/work/cuda-9.0/extras/CUPTI/lib64/:$LD_LIBRARY_PATH
    # ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/lib/x86_64-linux-gnu/libnccl.so
    rm /etc/apt/sources.list
    cp ${all_path}/sources.list /etc/apt
    apt-get update
    apt-get install libmysqlclient-dev -y
    apt-get install git -y
    apt-get install curl -y
    pip install MySQL-python


    save_log_dir=${all_path}/logs/log_${paddle_version}/${implement_type}

    if [[ -d ${save_log_dir} ]]; then
        rm -rf ${save_log_dir}
    fi
    # this for update the log_path coding mat
    export TRAIN_LOG_DIR=${save_log_dir}/train_log
    export PROFILER_LOG_DIR=${save_log_dir}/profiler_log
    mkdir -p ${TRAIN_LOG_DIR}
    mkdir -p ${PROFILER_LOG_DIR}

    train_log_dir=${save_log_dir}/train_log
    # mkdir -p ${train_log_dir}

    export ROOT_PATH=/home/crim
    export BENCHMARK_ROOT=${ROOT_PATH}/benchmark
    log_path=${BENCHMARK_ROOT}/${implement_type}/logs
    data_path=${all_path}/dataset
    prepare_path=${all_path}/prepare

    if [[ -e ${ROOT_PATH} ]]
    then
        rm ${log_path}/*
        cd ${ROOT_PATH}
        git pull
        echo "prepare had done"
    else
        mkdir /home/crim
        cd ${ROOT_PATH}
        git clone https://github.com/PaddlePaddle/benchmark.git
        cd ${BENCHMARK_ROOT}
        git submodule init
        git submodule update
        mkdir -p ${log_path}
    fi

    cd ${BENCHMARK_ROOT}
    benchmark_commit_id=$(git log|head -n1|awk '{print $2}')
    echo "benchmark_commit_id is: "${benchmark_commit_id}
    pip uninstall paddlepaddle-gpu -y
    pip install ${image_name}
    echo "*******prepare end!***********"
}

function run(){
    export ${implement_type}
    if [ ${implement_type} == "static_graph" ]; then
      source ${BENCHMARK_ROOT}/scripts/static_graph_models.sh
    else
      source ${BENCHMARK_ROOT}/scripts/dynamic_graph_models.sh
    fi

    if [ $model = "all" ]
    then
        for model_name in ${cur_model_list[@]}
        do
            echo "=====================${model_name} run begin=================="
            $model_name
            sleep 60
            echo "*********************${model_name} run end!!******************"
        done
    elif echo ${cur_model_list[@]} | grep -w $model &>/dev/null
    then
        echo "=====================${model} run begin=================="
        $model
        sleep 60
        echo "*********************${model} run end!!******************"
    else
        echo "model: $model not support!"
    fi
}


function save(){
    unset http_proxy
    unset https_proxy
    mv ${log_path} ${save_log_dir}/index
    ln -s ${all_path}/env/bin/python /usr/local/bin/mypython
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${all_path}/env/lib/
    cd ${origin_path}

    echo "==================== begin insert to sql ================="
    echo "benchmark_commit_id = ${benchmark_commit_id}"
    echo "   paddle_commit_id = ${image_commit_id}"
    echo "      paddle_branch = ${image_branch}"
    echo "     implement_type = ${implement_type}"
    echo "     paddle_version = ${paddle_version}"
    echo "       cuda_version = ${cuda_version}"
    echo "           log_path = ${save_log_dir}"
    echo "           job_type = ${job_type}"
    echo "        device_type = ${device_type}"

    mypython save.py --code_commit_id ${benchmark_commit_id} \
                 --image_commit_id ${image_commit_id} \
                 --image_branch ${image_branch} \
                 --log_path ${save_log_dir} \
                 --cuda_version ${cuda_version} \
                 --paddle_version ${paddle_version} \
                 --job_type ${job_type} \
                 --device_type ${device_type} \
                 --implement_type ${implement_type}

    echo "******************** end insert to sql!! *****************"
}

prepare
run
save

