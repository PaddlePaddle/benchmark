#!bin/bash

set -xe
if [[ $# -lt 1 ]]; then
    echo "running job dict is {1: speed, 2:mem, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 1|2|3 1(max_epoch)"
    exit
fi

function _set_params(){
    index=$1
    base_batch_size=20
    model_name="ptb_medium"_bs${base_batch_size}

    run_mode="sp" # Don't support mp
    max_epoch=${2}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi
 
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}

    direction_id=1
    mission_name="语言模型"
    skip_steps=15
    keyword="ips:"
    model_mode=-1
    ips_unit="words/s"

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
    grep -q "#To address val" train.py  # 模型暂不支持传参修改interval 和 关闭val
    if [ $? -eq 0 ]; then
        echo "----------already addressed val"
    else
        sed -i "s/eval_data=valid_loader,/\#eval_data=valid_loader, #To address val/g" train.py
        sed -i "s/\ \/\/\ 10/\ \/\/\ 30/g"  train.py
    fi

    train_cmd="--data_path ./data/simple-examples/data/ \
               --max_epoch ${max_epoch} \
               --device gpu"
    timeout 15m python -u train.py ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
        export job_fail_flag=1
    else
        echo -e "${model_name}, SUCCESS"
        export job_fail_flag=0
    fi
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_run
