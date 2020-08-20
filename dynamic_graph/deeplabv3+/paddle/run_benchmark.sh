#!bin/bash
set -xe
if [[ $# -lt 1 ]]; then
    echo "running job dict is {1: speed, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 1|3|6 sp|mp 600(max_epoch)"
    exit
fi

function _set_params(){
    index=$1
    base_batch_size=2
    model_name="deeplab"

    run_mode=${2:-"sp"} # Use sp for single GPU and mp for multiple GPU.
    max_epoch=${3:-"1"}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi
 
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

    log_file=${run_log_path}/dynamic_${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/dynamic_${model_name}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_dynamic_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}
}

function _train(){
    export PYTHONPATH=$(pwd):{PYTHONPATH}
    train_cmd="--model_name deeplabv3p_resnet50_vd_os8 \
               --dataset Cityscapes \
               --dataset_root dataset/cityscapes \
               --input_size 769 769 \
               --num_epochs ${max_epoch} \
               --save_interval_epochs 10 \
               --do_eval \
               --save_dir output_deeplabv3p_resnet50_vd \
               --batch_size ${base_batch_size} \
               --learning_rate 0.01 \
               --use_vdl \
               --num_workers 2 \
               --log_step 50"
    if [ ${run_mode} = "sp" ]; then
        train_cmd="python -u dygraph/benchmark/deeplabv3p.py "${train_cmd}
    else
        rm -rf ./mylog
        train_cmd="python -m paddle.distributed.launch --selected_gpus=$CUDA_VISIBLE_DEVICES --log_dir ./mylog dygraph/benchmark/deeplabv3p.py  --use_data_parallel=1 "${train_cmd}
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
