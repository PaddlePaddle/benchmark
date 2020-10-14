#!bin/bash

set -xe
if [[ $# -lt 1 ]]; then
    echo "running job dict is {1: speed, 2:mem, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 1|2|3 run_mode(sp|mp)"
    exit
fi

function _set_params(){
    index=$1
    run_mode=$2
    base_batch_size=8
    model_name="WaveNet"

    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi
 
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}
    
    direction_id=1
    mission_name="语音合成"
    skip_steps=5
    keyword="ips: "
    separator=" "
    position=13 #18
    model_mode=0 #1 

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}
    batch_size=`expr ${base_batch_size} \* ${num_gpu_devices}`

    log_file=${run_log_path}/dynamic_${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/dynamic_${model_name}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_dynamic_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}
}

function _train(){
    rm -rf output
    train_cmd="--config=../configs/wavenet_single_gaussian.yaml
               --data=./ljspeech
               --world_size=${num_gpu_devices}
               output"
    train_cmd="python -u train.py "${train_cmd}
    ${train_cmd} > ${log_file} 2>&1
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_run
