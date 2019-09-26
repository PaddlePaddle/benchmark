#!/bin/bash

set -xe

if [ $# -lt 3 ]; then
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_examples.sh speed|mem|maxbs 32 resnet50|resnet101 sp|mp /ssd1/ljh/logs"
    exit
fi

if [ "${BENCHMAKR_ROOT}" == "" ]; then
    export BENCHMARK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../../.." && pwd )"
fi

function _set_params() {
    index=$1
    base_batch_size=$2
    model_name=$3 # resnet50, resnet101
    run_mode=${4:-"sp"}
    run_log_root=${5:-$(pwd)}

    skip_steps=2
    keyword="Time"
    separator=" "
    position=-1
    range=5
    model_mode=0 # s/step -> samples/s

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    if [ $run_mode = "sp" ]; then
        batch_size=`expr $base_batch_size \* $num_gpu_devices`
    else
        batch_size=$base_batch_size
    fi
    num_workers=`expr 8 \* $num_gpu_devices`

    if [[ ${index} = "analysis" ]]; then
        log_file=${run_log_root}/log_examples_${model_name}_speed_${num_gpu_devices}_${run_mode}
    else
        log_file=${run_log_root}/log_examples_${model_name}_${index}_${num_gpu_devices}_${run_mode}
    fi
    log_parse_file=${log_file}
}

function _set_env() {
    echo "nothing"
}

function _train() {
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
    data_path=/data/ILSVRC2012/
    num_epochs=2

    python -c "import torch; print(torch.__version__)"
    export PYTHONPATH=${BENCHMARK_ROOT}/third_party/pytorch/vision

    echo "${model_name}, batch_size: ${batch_size}"
    python ${BENCHMARK_ROOT}/third_party/pytorch/examples/imagenet/main.py \
         --arch ${model_name} \
         --workers ${num_workers} \
         --epochs ${num_epochs} \
         --batch-size ${batch_size} \
         --resume ./output/examples \
         --print-freq 10 \
         --gpu ${CUDA_VISIBLE_DEVICES} \
         ${data_path} > ${log_file} 2>&1 &
  
    train_pid=$!
    sleep 300
    kill -9 `ps -ef|grep python |awk '{print $2}'`
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_set_env
_run
