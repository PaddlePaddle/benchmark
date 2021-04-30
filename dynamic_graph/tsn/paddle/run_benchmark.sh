#!bin/bash

set -x
if [[ $# -lt 1 ]]; then
    echo "running job dict is {1: speed, 2:mem, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 1|2|3 batch_size model_name(TSN) sp|mp 1(max_epoch)"
    exit
fi

function _set_params(){
    index=$1
    base_batch_size=${2}
    model_name=${3}_${base_batch_size}

    run_mode=${4:-"sp"} # Use sp for single GPU and mp for multiple GPU.
    max_epoch=${5:-"1"}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi
 
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}

    mission_name="视频分类"
    direction_id=0
    skip_steps=5
    keyword="ips:"
    model_mode=-1 
    ips_unit="images/s"

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    log_file=${run_log_path}/dynamic_${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/dynamic_${model_name}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_dynamic_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}
    batch_size=`expr ${base_batch_size} \* ${num_gpu_devices}`
}

function _train(){
    if [ ${run_mode} == "sp" ]; then
        config_files="./single_tsn_frame.yaml"
    elif [ ${run_mode} == "mp" ]; then
        config_files="./multi_tsn_frame.yaml"
        sed -i "s/learning_rate: 0.001/learning_rate: 0.002/g" ${config_files} # RD 暂未支持传LR
    else
        echo "------not support"
        exit
    fi

    train_cmd="--epoch ${max_epoch} \
               --batch_size=${batch_size} \
               --config=${config_files} \
               --log_interval=2 \
               --pretrain=./ResNet50_pretrained \
               --use_gpu=True \
               --validate=False \
               "
    if [ ${run_mode} = "sp" ]; then
        train_cmd="python -u train.py --use_data_parallel=False "${train_cmd}
    else
        rm -rf ./mylog
        train_cmd="python -m paddle.distributed.launch --log_dir ./mylog train.py --use_data_parallel=True "${train_cmd}
        log_parse_file="mylog/workerlog.0"
    fi
    
    timeout 15m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
        export job_fail_flag=1
    else
        echo -e "${model_name}, SUCCESS"
        export job_fail_flag=0
    fi
    if [ ${run_mode} != "sp"  -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_run
