#!bin/bash

set -xe
if [[ $# -lt 1 ]]; then
    echo "running job dict is {1: speed, 2:mem, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 1|2|3 800(max_iter)"
    exit
fi

function _set_params(){
    index=$1
    base_batch_size=20
    model_name="ptb"

    run_mode="sp" # Don't support mp
    max_iter=${2}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi
 
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}

    direction_id=1
    mission_name="语言模型"
    skip_steps=5
    keyword="batch_cost:"
    separator=" "
    position=10
    model_mode=2 # s/step -> steps/s

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    log_file=${run_log_path}/dynamic_${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/dynamic_${model_name}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_dynamic_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}
}

function _train(){
    train_cmd="--data_path  ./data/simple-examples/data/ \
               --max_iter ${max_iter} \
               --model_type small"
    python -u ptb_dy.py ${train_cmd} > ${log_file} 2>&1
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_run
