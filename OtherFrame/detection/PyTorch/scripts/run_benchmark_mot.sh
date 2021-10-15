#!/usr/bin/env bash
set -xe
# Usage：CUDA_VISIBLE_DEVICES=0 bash run_benchmark_mot.sh ${run_mode} ${batch_size} ${fp_item} ${max_epoch} ${model_name}

function _set_params(){
    run_mode=${1:-"sp"}            # sp|mp
    batch_size=${2:-"2"}
    fp_item=${3:-"fp32"}           # fp32|fp16
    max_epoch=${4:-"1"}
    model_name=${5:-"model_name"}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}  # TRAIN_LOG_DIR

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    log_file=${run_log_path}/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}
    res_log_file=${run_log_path}/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}_speed
}

function _analysis_log(){
    python analysis_log_mot.py \
                --filename ${log_file} \
                --jsonname ${res_log_file} \
                --keyword "time:" \
                --model_name detection_${model_name}_bs${batch_size}_${fp_item} \
                --run_mode ${run_mode} \
                --gpu_num ${num_gpu_devices} \
                --batch_size ${batch_size}
    cp ${res_log_file} /workspace
}
