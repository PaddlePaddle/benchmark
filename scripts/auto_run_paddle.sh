#!/bin/bash

function usage () {
  cat <<EOF
  usage: $0 [options]
  -h         optional   Print this help message
  -m  model  run models
  -c  cuda_version 11.0|10.1|11.2
  -d  cudnn_version 7|8
  -n  image_name
  -i  image_commit_id
  -a  image_branch develop|1.6|pr_number|v1.6.0
  -v  paddle_version
  -p  all_path contains dir of prepare(pretrained models), dataset, logs, such as /ssd1/ljh
  -t  job_type  benchmark_daliy | models test | pr_test
  -g  device_type  A100 | v100
  -s  implement_type of model static_graph | dynamic_graph | dynamic_to_static
EOF
}
if [ $# -lt 18 ] ; then
  usage
  exit 1;
fi
while getopts h:m:c:n:i:a:v:p:d:t:g:s: opt
do
  case $opt in
  h) usage; exit 0 ;;
  m) model="$OPTARG" ;;
  c) cuda_version="$OPTARG" ;;
  d) cudnn_version="$OPTARG" ;;
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

    # this is for image paddlepaddle/paddle:latest-gpu-cuda${cuda_version}-cudnn${cudnn_version}
    #export LD_LIBRARY_PATH=/home/work/418.39/lib64/:$LD_LIBRARY_PATH       # fixed nvidia-docker, remove temporarily 

    # NOTE: this path is for profiler
    # export LD_LIBRARY_PATH=/home/work/cuda-9.0/extras/CUPTI/lib64/:$LD_LIBRARY_PATH
    # ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/lib/x86_64-linux-gnu/libnccl.so
    rm /etc/apt/sources.list
    cp ${all_path}/sources.list /etc/apt
    apt-get update
    apt-get install libmysqlclient20=5.7.33-0ubuntu0.16.04.1 --allow-downgrades
    apt-get install libmysqlclient-dev git curl psmisc -y
    pip install MySQL-python


    save_log_dir=${all_path}/logs/${paddle_version}/${implement_type}

    if [[ -d ${save_log_dir} ]]; then
        rm -rf ${save_log_dir}
    fi
    # this for update the log_path coding mat
    export TRAIN_LOG_DIR=${save_log_dir}/train_log
    export PROFILER_LOG_DIR=${save_log_dir}/profiler_log
    export IMPLEMENT_TYPE=${implement_type}
    
    mkdir -p ${TRAIN_LOG_DIR}
    mkdir -p ${PROFILER_LOG_DIR}

    train_log_dir=${save_log_dir}/train_log
    # mkdir -p ${train_log_dir}

    export ROOT_PATH=/home/crim
    export BENCHMARK_ROOT=${ROOT_PATH}/benchmark
    log_path=${BENCHMARK_ROOT}/${implement_type}/logs
    data_path=${all_path}/dataset
    prepare_path=${all_path}/prepare

    # 每个任务每个模式均做创建处理，并删除上一次任务的残存文件，避免相同repo不通分支引入的bug
    mkdir -p ${ROOT_PATH}
    cd ${ROOT_PATH}
    rm -rf *
    git clone https://github.com/PaddlePaddle/benchmark.git --recursive
    mkdir -p ${log_path}
    echo "****************${implement_type} prepare had done*****************"

    cd ${BENCHMARK_ROOT}
    benchmark_commit_id=$(git log|head -n1|awk '{print $2}')
    echo "benchmark_commit_id is: "${benchmark_commit_id}

    # 动态图升级到cuda10.1 python3.7，静态图切cuda10.1 python3.7
    if [[ 'dynamic_graph' == ${implement_type} ]] || [[ 'static_graph' == ${implement_type} ]] || [[ 'dynamic_to_static' == ${implement_type} ]]; then
        rm -rf run_env
        mkdir run_env
        ln -s $(which python3.7) run_env/python
        ln -s $(which pip3.7) run_env/pip
        export PATH=$(pwd)/run_env:${PATH}
        pip install -U pip
        echo `pip --version`
    fi
    pip uninstall paddlepaddle-gpu -y
    pip install ${image_name}
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
    if [ $? -ne 0 ]; then
        pip install ${all_path}/tools/opencv_python-4.5.1.48-cp37-cp37m-manylinux2014_x86_64.whl
    fi

    # dali install
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda$(echo ${cuda_version}|cut -d "." -f1)0    # note: dali 版本格式是cuda100 & cuda110

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
    export ${implement_type}
    if [ ${implement_type} == "static_graph" ]; then
      source ${BENCHMARK_ROOT}/scripts/static_graph_models.sh
    elif [ ${implement_type} == "dynamic_graph" ]; then
      source ${BENCHMARK_ROOT}/scripts/dynamic_graph_models.sh
    else
      source ${BENCHMARK_ROOT}/scripts/dynamic_to_static_models.sh
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
    echo "       cudnn_version = ${cudnn_version}"
    echo "           log_path = ${save_log_dir}"
    echo "           job_type = ${job_type}"
    echo "        device_type = ${device_type}"

    mypython -m pip install PyMySQL==0.10.1
    mypython save.py --code_commit_id ${benchmark_commit_id} \
                 --image_commit_id ${image_commit_id} \
                 --image_branch ${image_branch} \
                 --log_path ${save_log_dir} \
                 --cuda_version ${cuda_version} \
                 --cudnn_version ${cudnn_version} \
                 --paddle_version ${paddle_version} \
                 --job_type ${job_type} \
                 --device_type ${device_type} \
                 --implement_type ${implement_type}

    echo "******************** end insert to sql!! *****************"
}

prepare
run
save

