#!bin/bash

set -xe

if [ $# -lt 1 ]; then
    echo "Usage: "
    echo "CUDA_VISIBLE_DEVICES=0 bash run_pix2pix.sh speed  /ssd2/liyang/logs"
    exit
fi

function _set_params() {
    index=$1
    run_log_root=${2:-$(pwd)}
    skip_steps=1
    keyword="time:"
    run_mode="sp"
    separator=" "
    position=5
    range=5
    model_mode=2
    model_name=pix2pix
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}
    base_batch_size=1

    log_file=${run_log_root}/log_${model_name}_speed_${num_gpu_devices}_${run_mode}
}

function _train() {
  echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size}"
  env no_proxy=localhost python3.6 train.py \
              --dataroot ./dataset \
              --name cityscapes_pix2pix \
              --model pix2pix \
              --netG unet_256     \
              --direction BtoA    \
              --lambda_L1 100     \
              --dataset_mode aligned \
              --norm batch        \
              --batch_size=${base_batch_size}    \
              --pool_size 0   > ${log_file} 2>&1 &
  train_pid=$!
  sleep 200
  kill -9 $train_pid
}

source ${BENCHMARK_ROOT}/competitive_products/common_scripts/run_model.sh
_set_params $@
_run
