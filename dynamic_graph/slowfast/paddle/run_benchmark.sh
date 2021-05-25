#!bin/bash

set -xe
if [[ $# -lt 1 ]]; then
    echo "running job dict is {1: speed, 2:mem, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 1|2|3 sp|mp 1(max_epoch)"
    exit
fi

function _set_params(){
    index=$1
    base_batch_size=8
    model_name="SlowFast"_bs${base_batch_size}

    run_mode=${2:-"sp"} # Use sp for single GPU and mp for multiple GPU.
    max_epoch=${3:-"1"}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi
 
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}

    mission_name="视频分类"
    direction_id=0
    skip_steps=10
    keyword="ips:"
    model_mode=-1
    ips_unit="instance/s"

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    log_file=${run_log_path}/dynamic_${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/dynamic_${model_name}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_dynamic_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}
#    batch_size=`expr $base_batch_size \* $num_gpu_devices`
}

function _train(){
    train_cmd="-c configs/recognition/slowfast/slowfast.yaml
               -o epochs=${max_epoch}
               -o DATASET.num_workers=8
               -o DATASET.batch_size=${base_batch_size}
               -o DATASET.train.file_path=./data/train.csv
               -o DATASET.valid.file_path=./data/val.csv
               -o DATASET.test.file_path=./data/val.csv
               -o log_interval=2
               "

    if [ ${run_mode} = "sp" ]; then
        train_cmd="python -B -u main.py "${train_cmd}
    else
        rm -rf ./mylog
        train_cmd="python -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir ./mylog main.py "${train_cmd}
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
