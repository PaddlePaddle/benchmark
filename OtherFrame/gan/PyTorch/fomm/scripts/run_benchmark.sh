#!/usr/bin/env bash
set -xe

# Test training benchmark for a model.

# Usage: CUDA_VISIBLE_DEVICES=xxx bash run_benchmark.sh ${model_item} ${run_mode} ${fp_item} ${bs_item} ${max_iter} ${num_workers}

function _set_params(){
    model_item=${1:-"model_item"}
    run_mode=${2:-"sp"}         # sp or mp
    fp_item=${3:-"fp32"}        # fp32 or fp16
    batch_size=${4:-"2"}
    max_iter=${5:-"100"}
    num_workers=${6:-"3"}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    log_file=${run_log_path}/${model_item}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}
    res_log_file=${run_log_path}/${model_item}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}_speed
    model_name=${model_item}_bs${batch_size}_${fp_item}
}

function _analysis_log(){
    python analysis_log.py ${model_name} ${log_file} ${res_log_file}
    cp ${res_log_file} /workspace
}

function _train(){
    echo "Train ${model_name} on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"

    case ${run_mode} in
    sp) train_cmd="python run.py --config config/vox_256_sp_${batch_size}.yaml " ;;
    mp)
        train_cmd="python run.py --config config/vox_256_mp_${batch_size}.yaml  --device_ids $CUDA_VISIBLE_DEVICES" ;;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac

    timeout 15m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
        export job_fail_flag=1
    else
        echo -e "${model_name}, SUCCESS"
        export job_fail_flag=0
    fi
    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi

    _analysis_log

    trap 'for pid in $(jobs -pr); do kill -KILL $pid; done' INT QUIT TERM
}

_set_params $@
_train
