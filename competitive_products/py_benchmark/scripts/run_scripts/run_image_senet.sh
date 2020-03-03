#!/bin/bash

set -x

if [ $# -lt 3 ]; then
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_senet.sh speed|mem|maxbs 32 se_resnext_50|resnet_50|resnet_101 sp|mp /ssd1/ljh/logs"
    exit
fi

function _set_params() {
    index=$1
    base_batch_size=$2
    model_name=$3 # se_resnext_50, resnet_50, resnet_101
    run_mode="sp" # Multi-processes is not supported
    run_log_root=${5:-$(pwd)}

    skip_steps=2
    keyword="Time:"
    separator=" "
    position=11
    range=5
    model_mode=3 # steps/s -> samples/s

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    batch_size=`expr $base_batch_size \* $num_gpu_devices`
    num_workers=`expr 8 \* $num_gpu_devices`

    if [[ ${index} = "analysis" ]]; then
        log_file=${run_log_root}/log_senet_${model_name}_speed_${num_gpu_devices}_${run_mode}
    else
        log_file=${run_log_root}/log_senet_${model_name}_${index}_${num_gpu_devices}_${run_mode}
    fi
    log_parse_file=${log_file}
}

function _set_env() {
    echo "nothing ..."
}

function _train() {
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
    WORK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")" && pwd )"
    num_epochs=2

    python -c "import torch; print(torch.__version__)"
    export PYTHONPATH=${BENCHMARK_ROOT}/third_party/pytorch/vision

    echo "${model_name}, batch_size: ${batch_size}"
    cd ./SENet/
    stdbuf -oL python train.py \
        --network ${model_name} \
        --data-dir ImageData/ \
        --batch-size ${batch_size} \
        --num-workers ${num_workers} \
        --num-epochs ${num_epochs} \
        --gpus ${CUDA_VISIBLE_DEVICES} > ${log_file} 2>&1 &

    train_pid=$!
    sleep 300
    kill -9 `ps -ef|grep python |awk '{print $2}'`
    cd ${WORK_ROOT}
}

source ${PYTORCH_BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_set_env
_run
