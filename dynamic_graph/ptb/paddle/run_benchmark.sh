#!bin/bash
set -xe
if [[ $# -lt 1 ]]; then
    echo "running job dict is {1: speed, 2:mem, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1|2|3 sp|mp 8000(max_iter)"
    exit
fi

function _set_params(){
    index=$1
    base_batch_size=20
    model_name="ptb"

    run_mode="sp" # Don't support mp
    max_iter=${3}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi
 
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}

    direction_id=1
    mission_name="语言模型"
    skip_steps=0
    keyword="batch cost"
    separator=" "
    position=11
    model_mode=2 # s/step -> steps/s

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
    profiler_path=${profiler_path}/profiler_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}
}

function _set_env(){
    #开启gc
    echo "nothing"
}

function _train(){
   train_cmd="--data_path  ./data/simple-examples/data/ \
              --max_iter ${max_iter} \
              --model_type small"

    python -u ptb_dy.py ${train_cmd} > ${log_file} 2>&1
    kill -9 `ps -ef|grep python |awk '{print $2}'`
#    if [ ${num_gpu_devices} -eq 1 ]; then
#        train_cmd="python -u ptb_dy.py "${train_cmd}
#    else
#        rm ./mylog
#        train_cmd="python3 -m paddle.distributed.launch --log_dir=./mylog ptb_dy.py "${train_cmd}
#        log_parse_file="mylog/workerlog.0"
#    fi
#    ${train_cmd} > ${log_file} 2>&1
#    kill -9 `ps -ef|grep python |awk '{print $2}'`
#    if [ ${num_gpu_devices} != 1  -a -d mylog ]; then
#        rm ${log_file}
#        cp mylog/workerlog.0 ${log_file}
#    fi


    
#--use_data_parallel=1
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_set_env
_run
