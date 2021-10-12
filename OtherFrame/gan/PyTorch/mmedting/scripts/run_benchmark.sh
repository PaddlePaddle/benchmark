#!/usr/bin/env bash
set -xe

# Test training benchmark for a model.

# Usage: CUDA_VISIBLE_DEVICES=xxx bash run_benchmark.sh ${model_name} ${run_mode} ${fp_item} ${bs_item} ${max_iter} ${num_workers}

function _set_params(){
    model_name=${1:-"model_name"}
    run_mode=${2:-"sp"}         # sp or mp
    fp_item=${3:-"fp32"}        # fp32 or fp16
    batch_size=${4:-"2"}
    max_iter=${5:-"100"}
    num_workers=${6:-"3"}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    log_file=${run_log_path}/${model_name}_${fp_item}_${num_gpu_devices}
    res_log_file=${run_log_path}/${model_name}_${fp_item}_${num_gpu_devices}_speed
}

function _analysis_log(){
    python analysis_log.py ${model_name} ${batch_size} ${log_file} ${res_log_file}
    cp ${log_file} /workspace
    cp ${res_log_file} /workspace
}

function _train(){
    echo "Train ${model_name} on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"

    train_config="mmedi_benchmark_configs/${model_name}.py"
    train_options="--no-validate "

    case ${run_mode} in
    sp) train_cmd="./tools/dist_train.sh ${train_config} 1 ${train_options}" ;;
    mp)
        case ${model_name} in
        basicvsr_mp_bs2|basicvsr_mp_bs4) train_cmd="./tools/dist_train.sh ${train_config} 4 ${train_options}" ;;
        *) train_cmd="./tools/dist_train.sh ${train_config} 8 ${train_options}"
        esac
        ;;
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

    kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
}

_set_params $@
_train