#!/usr/bin/env bash
set -xe
# 运行示例：CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} ${model_mode} ${config_path}
# 参数说明
function _set_params(){
    run_mode=${1:-"sp"}          # 单卡sp|多卡mp
    batch_size=${2:-"64"}
    fp_item=${3:-"fp32"}        # fp32|fp16
    model_name=${4:-"model_name"}
    config_path=${5:-"config_path"}
    run_log_path="${TRAIN_LOG_DIR:-$(pwd)}"  # TRAIN_LOG_DIR 后续QA设置该参
 
#   以下不用修改   
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    log_file=${run_log_path}/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}.log
}
function _train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"

    case ${run_mode} in
    sp) train_cmd="python tools/train.py ${config_path} --no-validate";;
    mp)
        train_cmd="bash ./tools/dist_train.sh ${config_path} ${num_gpu_devices}";;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac

# 以下不用修改
    timeout 10m ${train_cmd}
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
        export job_fail_flag=1
    else
        echo -e "${model_name}, SUCCESS"
        export job_fail_flag=0
    fi
    kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
 
}

_set_params $@
_train
