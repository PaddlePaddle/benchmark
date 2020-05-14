#!bin/bash
set -xe

if [[ $# -lt 4 ]]; then
    echo "running job dict is {1: speed, 2:mem, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1|2|3 sp|mp 1000(max_iter) model_name(MobileNetV1|MobileNetV2)"
    exit
fi


function _set_params(){
    index=$1
    base_batch_size=256
    if [ ${4} != "MobileNetV1" ] && [ ${4} != "MobileNetV2" ]; then
        echo "------------> please check the model name!"
        exit 1
    fi
    model_name=${4}

    run_mode="sp" # Don't support mp
    max_iter=${3}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi
 
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}

    direction_id=0
    mission_name="图像分类"
    skip_steps=10
    keyword="net_t:"
    separator=" "
    position=13
    #range=0:9
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

function _train(){
   train_cmd="--batch_size=${batch_size} \
              --total_images=1281167 \
              --class_dim=1000 \
              --image_shape=3,224,224 \
              --model_save_dir=output \
              --lr_strategy=piecewise_decay \
              --lr=0.1 \
              --data_dir=./data/ILSVRC2012 \
              --l2_decay=3e-5 \
              --model=${model_name} \
              --max_iter=${max_iter} \
              --num_epochs=2 "
#              --is_profiler=${is_profiler} \
#              --profiler_path=${profiler_path} \
    if [ ${num_gpu_devices} -eq 1 ]; then
        train_cmd="python -u train.py "${train_cmd}
    else
        rm -r ./mylog
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog train.py --use_data_parallel=1 "${train_cmd}
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
