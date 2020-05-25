#!bin/bash
set -xe
if [[ $# -lt 1 ]]; then
    echo "running job dict is {1: speed, 2:mem, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1|2|3 sp|mp 600(max_iter)"
    exit
fi

function _set_params(){
    index=$1
    base_batch_size=32
    model_name="resnet"

    run_mode="sp" # Don't support mp
    max_iter=${3}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi
 
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}

    mission_name="图像分类"
    direction_id=0
    skip_steps=5
    keyword="batch cost:"
    separator=" "
    position=14
    model_mode=0 # s/step -> steps/s

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    if [[ ${run_mode} = "sp" ]]; then
        batch_size=`expr $base_batch_size \* $num_gpu_devices`
    else
        batch_size=$base_batch_size
    fi

    log_file=${run_log_path}/dynamic_${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/dynamic_${model_name}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_dynamic_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}
}

function _set_env(){
    #开启gc
    echo "nothing"
}

echo ${CUDA_VISIBLE_DEVICES}


function _train(){
    train_cmd="--max_iter ${max_iter} \
              --batch_size=${batch_size}
              "
#              --data_dir=./ILSVRC2012_Pytorch/dataset_100/  
#              --model=ResNet50
#              --is_profiler=${is_profiler} \
#              --profiler_path=${profiler_path} \
    if [ ${num_gpu_devices} -eq 1 ]; then
        train_cmd="python -u train.py "${train_cmd}
    else
        rm -rf ./mylog
        train_cmd="python -m paddle.distributed.launch --selected_gpus=$CUDA_VISIBLE_DEVICES  --log_dir ./mylog train.py   --use_data_parallel=1 "${train_cmd}
        log_parse_file="mylog/workerlog.0"
    fi
    
    ${train_cmd} > ${log_file} 2>&1
    kill -9 `ps -ef|grep python |awk '{print $2}'`
    if [ ${num_gpu_devices} != 1  -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi

}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_set_env
_run
