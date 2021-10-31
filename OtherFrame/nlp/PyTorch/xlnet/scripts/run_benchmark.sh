#!/usr/bin/env bash
set -xe

# Test training benchmark for a model.
# 运行示例：CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ${run_mode} ${batch_size} ${fp_item} 1500 ${model_mode}


function _set_params(){
    run_mode=${1:-"sp"}          # 单卡sp|多卡mp
    batch_size=${2:-"64"}
    fp_item=${3:-"fp32"}        # fp32|fp16
    max_iter=${4:-"1500"}       # 可选，如果需要修改代码提前中断
    model_name=${5:-"xlnet-base-cased"}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}  # TRAIN_LOG_DIR 后续QA设置该参数

#   以下不用修改
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    log_file=${run_log_path}/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}
}


function _train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"

    train_cmd="--model_name_or_path=${model_name}
               --logging_dir=${run_log_path}
               --task_name=sst2
               --max_seq_length=128
               --per_device_train_batch_size=${batch_size}
               --learning_rate=2e-5
               --num_train_epochs=3
               --max_steps=1500
               --pad_to_max_length=False
               --logging_strategy=steps
               --logging_steps=500
               --do_train
               "

    case ${run_mode} in
    sp)
        train_cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python run_glue.py ${train_cmd}" ;;
    mp)
        train_cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python run_glue.py ${train_cmd}" ;;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac

    # 以下不用修改
    timeout 15m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
        export job_fail_flag=1
    else
        echo -e "${model_name}, SUCCESS"
        export job_fail_flag=0
    fi
    kill -9 `ps -ef|grep 'python'|awk '{print $2}'`

    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi

    trap 'for pid in $(jobs -pr); do kill -KILL $pid; done' INT QUIT TERM
}

_set_params $@
_train
_analysis_log
