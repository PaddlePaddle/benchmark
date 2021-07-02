#!bin/bash

set -xe
if [[ $# -lt 1 ]]; then
    echo "running job dict is {1: speed, 2:mem, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 1|2|3 model_name(bow|lstm|bilstm|gru|bigru|rnn|birnn|cnn) run_mode(sp|mp) 1(max_epoch)"
    exit
fi

function _set_params(){
    index=$1
    base_batch_size=64
    net=$2
    model_name="Senta"_${net}_bs${base_batch_size}

    run_mode=$3
    max_epoch=${4}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi
 
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}
    
    direction_id=1
    mission_name="文本分类"
    skip_steps=5
    keyword="ips "
    separator=" "
    position=13 #18
    model_mode=0 #1
    range=1:6

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
    train_cmd="--use_gpu=True
               --network_name=${net}
               --checkpoint_dir ${model_name}_saved_model.param
               --num_epoch=${max_epoch}"
    if [ ${run_mode} = "sp" ]; then
        train_cmd="python -m paddle.distributed.launch  main.py "${train_cmd}
    else
        rm -rf ./mylog
        train_cmd="python -m paddle.distributed.launch   --log_dir=./mylog main.py "${train_cmd}
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
        cp mylog/`ls -l mylog/ | awk '/^[^d]/ {print $5,$9}' | sort -nr | head -1 | awk '{print $2}'` ${log_file}
    fi
    kill -9 `ps -ef|grep python |awk '{print $2}'`
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_run
