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

    mission_name="语音合成"
    direction_id=1
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    # 当前在 models 里面
    pushd ../../../
    run_log_path=$PWD/scripts/logs
    popd
    echo $run_log_path
    
    if [ ! -d $run_log_path/train_log ];then
        mkdir -p $run_log_path/train_log
    fi

    if [ ! -d $run_log_path/index ];then
        mkdir -p $run_log_path/index
    fi

    log_file=${run_log_path}/train_log/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}
    res_log_file=${run_log_path}/index/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}_speed
}

function _analysis_log(){
    python analysis_log.py ${log_file} ${res_log_file} ${mission_name} ${direction_id}
    cp ${res_log_file} ${BENCHMARK_ROOT}/
}

function _train(){
    echo "Train ${model_name} on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"

    train_cmd="--batch-size ${batch_size} --max-iter ${max_iter}"  

    case ${run_mode} in
    sp) train_cmd="bash run.sh --stage 2 --stop-stage 2 --n-gpus 1 ${train_cmd}" ;;
    mp)
        train_cmd="bash run.sh --stage 2 --stop-stage 2 --n-gpus 2 ${train_cmd}" ;;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac
    echo "$train_cmd"
    pushd ParallelWaveGAN/egs/csmsc/voc1
    timeout 15m ${train_cmd}
    mv exp/train_nodev_csmsc_parallel_wavegan.v1/train.log ${log_file}
    popd
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
