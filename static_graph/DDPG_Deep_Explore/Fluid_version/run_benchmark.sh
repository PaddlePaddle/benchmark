#!bin/bash
set -xe

if [[ $# -lt 1 ]]; then
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh speed|mem sp|mp /ssd1/ljh/logs"
    exit
fi

function _set_params(){
    index="$1"
    run_mode=${2:-"sp"}
    run_log_path=${3:-$(pwd)}

    model_name="ddpg_deep_explore"
    skip_steps=1
    keyword="time consuming"
    separator=" "
    position=9
    model_mode=1

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}
    base_batch_size=1
    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_parse_file=${log_file}
}

function _set_env(){
    #打开后速度变快
    export FLAGS_cudnn_exhaustive_search=1

    #开启
    export FLAGS_eager_delete_tensor_gb=0.0
    export FLAGS_fast_eager_deletion_mode=1
}

function _train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
    for i in {1..5}
    do
        FLAGS_enforce_when_check_program_=0 GLOG_vmodule=operator=1,computation_op_handle=1 \
        python ./multi_thread_test.py \
            --ensemble_num 1 \
            --test_times 10 >> ${log_file} 2>&1
    done
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_set_env
_run