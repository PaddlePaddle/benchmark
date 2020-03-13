#!/bin/bash

set -x
#nvprof -o timeline_output_medium -f --cpu-profiling off  --profile-from-start off  python  train.py \
#export CUDA_VISIBLE_DEVICES=7
  echo "Usage: "
  echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh speed|mem large|medium|small static|padding /path/to/log"

function _set_params(){
    index=$1
    model_type=$2
    rnn_type=$3
    base_batch_size=20
    run_log_path=${4:-$(pwd)}

    skip_steps=3
    keyword="-- Epoch:"
    separator=" "
    position=4
    model_mode=2
    run_mode=sp

    devices_str=${CUDA_VISIBLE_DEVICES//,/ }
    gpu_devices=($devices_str)
    num_gpu_devices=${#gpu_devices[*]}
    batch_size=`expr ${base_batch_size} \* $num_gpu_devices`
    
    model_name=padding_${model_type}_${rnn_type}
    log_file=${run_log_path}/log_${index}_${model_name}_${num_gpu_devices}
}

function _train(){
  echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices"
  if [ ${index} = "speed" ]; then
      sed -i '93c \    config.gpu_options.allow_growth = False' train.py
  elif [ ${index} = "mem" ]; then
      echo "this index is: "$index
      sed -i '93c \    config.gpu_options.allow_growth = True' train.py
  fi
  python -u train.py \
    --model_type $model_type \
    --rnn_type $rnn_type > ${log_file} 2>&1 &
  train_pid=$!
  sleep 600
  kill -9 $train_pid
}

source ${BENCHMARK_ROOT}/competitive_products/common_scripts/run_model.sh
_set_params $@
_run

