#!bin/bash
set -xe

if [[ $# -lt 1 ]]; then
    echo "running job dict is {1: speed, 2:mem, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 1|2|3 sp|mp model_name(CycleGAN|Pix2pix) 1(max_epoch)"
    exit
fi

function _set_params(){
    index=$1
    base_batch_size=1
    run_mode=${2:-"sp"} # Use sp for single GPU and mp for multiple GPU.
    model_name=$3
    max_epoch=${4:-"1"}
    if [ ${3} != "CycleGAN" ] && [ ${3} != "Pix2pix" ]; then
        echo "Please check the model name! it should be CycleGAN|Pix2pix"
        exit 1
    fi

    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}

    mission_name="图像生成"
    direction_id=0
    keyword="ips:"
    skip_steps=5
    ips_unit="images/sec"

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
    export PYTHONPATH=$PWD:$PYTHONPATH
   
    # 暂不支持传入epochs，暂时用sed 的方式 
    sed -i "1c epochs: ${max_epoch}" configs/$(echo $model_name | tr '[A-Z]' '[a-z]')_cityscapes.yaml
   
    train_cmd="--config-file configs/$(echo $model_name | tr '[A-Z]' '[a-z]')_cityscapes.yaml"
    if [ ${run_mode} = "sp" ]; then
        train_cmd="python -u tools/main.py "${train_cmd}
    else
        rm -rf ./mylog
        train_cmd="python -m paddle.distributed.launch --selected_gpus=$CUDA_VISIBLE_DEVICES --log_dir ./mylog tools/main.py --use_data_parallel=1 "${train_cmd}
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
