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
    run_mode=${2:-"sp"} # Use sp for single GPU and mp for multiple GPU.
    max_epoch=${3:-"1"}
    base_batch_size=128
    model_name=${4}_bs${base_batch_size}
    if [ ${4} != "MobileNetV1" ] && [ ${4} != "MobileNetV2" ]; then
            echo "------------> please check the model name!"
            exit 1
    fi
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
    batch_size=${base_batch_size}

    log_file=${run_log_path}/dynamic_to_static_${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/dynamic_to_static_${model_name}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_dynamic_to_static_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}
}

function _train(){
    train_cmd="-c ./ppcls/configs/ImageNet/${model_name%_bs*}/${model_name%_bs*}.yaml
               -o Global.epochs=${max_epoch}
               -o Global.eval_during_train=False
               -o Global.save_interval=2
               -o Global.to_static=True
               -o DataLoader.Train.sampler.batch_size=${batch_size}
               -o DataLoader.Train.dataset.image_root=./dataset/imagenet100_data
               -o DataLoader.Train.dataset.cls_label_path=./dataset/imagenet100_data/train_list_ori.txt
               -o DataLoader.Train.loader.num_workers=8"
    if [ ${run_mode} = "sp" ]; then
        train_cmd="python -u tools/train.py "${train_cmd}
    else
        rm -rf ./mylog_${model_name}
        train_cmd="python -m paddle.distributed.launch --gpus=$CUDA_VISIBLE_DEVICES --log_dir ./mylog_${model_name} tools/train.py "${train_cmd}
        log_parse_file="mylog_${model_name}/workerlog.0"
    fi
    
    timeout 15m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
        export job_fail_flag=1
    else
        echo -e "${model_name}, SUCCESS"
        export job_fail_flag=0
    fi

    if [ ${run_mode} != "sp"  -a -d mylog_${model_name} ]; then
        rm ${log_file}
        cp mylog_${model_name}/`ls -l mylog_${model_name}/ | awk '/^[^d]/ {print $5,$9}' | sort -nr | head -1 | awk '{print $2}'` ${log_file}
    fi
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_run
