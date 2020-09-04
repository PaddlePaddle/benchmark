#!bin/bash
set -xe
if [[ $# -lt 1 ]]; then
    echo "running job dict is {1: speed, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 1|3|6 sp|mp model_item(HRnet|deeplabv3) 600(max_iter)"
    exit
fi

function _set_params(){
    index=$1
    base_batch_size=2
    run_mode=${2:-"sp"} # Use sp for single GPU and mp for multiple GPU.
    model_item=${3}
    max_iter=${4:-"200"}

    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}

    mission_name="图像分割"
    direction_id=0
    skip_steps=5
    keyword="batch_cost="
    separator="="
    position=5
    range=0:5
    model_mode=0 # s/step -> sample/s 

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    log_file=${run_log_path}/dynamic_${model_item}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/dynamic_${model_item}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_dynamic_${model_item}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}
}

function _train(){
    export PYTHONPATH=$(pwd):{PYTHONPATH}
    if [ ${model_item} = "HRnet" ]; then
        model_name="fcn_hrnet_w18"
        input_size="1024 512"
        model_script="hrnet.py"
    elif [ ${model_item} = "deeplabv3" ]; then
        model_name="deeplabv3p_resnet50_vd_os8"
        #model_name="deeplabv3p_resnet50_vd"
        input_size="769 769"
        model_script="deeplabv3p.py"
    else
        echo "------------------>model_item should be HRnet or deeplabv3!"
        exit 1
    fi

    train_cmd="--model_name ${model_name}
               --dataset Cityscapes
               --dataset_root ./cityscape 
               --input_size ${input_size}
               --iters ${max_iter} 
               --batch_size ${base_batch_size} 
               --learning_rate 0.01 
               --num_workers 2 
               --log_iters 5"

    if [ ${run_mode} = "sp" ]; then
        #train_cmd="python -m paddle.distributed.launch --selected_gpus=$CUDA_VISIBLE_DEVICES dygraph/benchmark/${model_script} "${train_cmd}
        train_cmd="python -u dygraph/benchmark/${model_script} "${train_cmd}
    else
        rm -rf ./mylog
        train_cmd="python -m paddle.distributed.launch --selected_gpus=$CUDA_VISIBLE_DEVICES --log_dir ./mylog dygraph/benchmark/${model_script} "${train_cmd}
        log_parse_file="mylog/workerlog.0"
    fi

    ${train_cmd} > ${log_file} 2>&1
    if [ ${run_mode} != "sp"  -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_run
