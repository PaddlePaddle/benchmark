#!bin/bash

set -xe
if [[ $# -lt 1 ]]; then
    echo "running job dict is {1: speed, 2:mem, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 1|3|6 sp|mp 1(max_epoch) model_name(MobileNetV1|MobileNetV2)" 
    exit
fi

function _set_params(){
    index=$1
    base_batch_size=128
    model_name=${4}
    if [ ${4} != "MobileNetV1" ] && [ ${4} != "MobileNetV2" ]; then
            echo "------------> please check the model name!"
            exit 1
    fi

    run_mode=${2:-"sp"} # Use sp for single GPU and mp for multiple GPU.
    max_epoch=${3:-"1"}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi
 
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}

    mission_name="图像分类"
    direction_id=0
    skip_steps=11
    keyword="ips:"
    model_mode=-1
    ips_unit="images/s"

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}
    batch_size=`expr ${num_gpu_devices} \* ${base_batch_size}`

    log_file=${run_log_path}/dynamic_${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/dynamic_${model_name}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_dynamic_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}
}

function _train(){
    train_cmd="-c ./configs/${model_name}/${model_name}.yaml 
               -o validate=False
               -o epochs=${max_epoch}
               -o print_interval=10
               -o TRAIN.batch_size=${batch_size}
               -o TRAIN.data_dir=./dataset/dataset_100
               -o TRAIN.file_list=./dataset/dataset_100/train_list_mobile.txt
               -o TRAIN.num_workers=8"
    if [ ${run_mode} = "sp" ]; then
        train_cmd="python -u tools/train.py "${train_cmd}
    else
        rm -rf ./mylog_${model_name}
        train_cmd="python -m paddle.distributed.launch --gpus=$CUDA_VISIBLE_DEVICES --log_dir ./mylog_${model_name} tools/train.py "${train_cmd}
        log_parse_file="mylog_${model_name}/workerlog.0"
    fi
    
    timeout 15m ${train_cmd} > ${log_file} 2>&1
    if [ ${run_mode} != "sp"  -a -d mylog_${model_name} ]; then
        rm ${log_file}
        cp mylog_${model_name}/`ls -l mylog_${model_name}/ | awk '/^[^d]/ {print $5,$9}' | sort -nr | head -1 | awk '{print $2}'` ${log_file}
    fi
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_run
