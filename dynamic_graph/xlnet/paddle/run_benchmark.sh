#!/usr/bin/env bash
set -xe

# 运行示例：CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ${run_mode} ${batch_size} ${fp_item} 1500 ${model_mode}
# 参数说明
function _set_params(){
    run_mode=${1:-"sp"}          # 单卡sp|多卡mp
    batch_size=${2:-"64"}
    fp_item=${3:-"fp32"}        # fp32|fp16
    max_iter=${4:-"200"}       # 可选，如果需要修改代码提前中断
    model_item=${5:-"xlnet-base-cased"}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}  # TRAIN_LOG_DIR 后续QA设置该参数
    need_profile=${6:-"off"}

    base_batch_size=${batch_size}
    mission_name="语义表示"
    direction_id=1

#   以下不用修改
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    log_file=${run_log_path}/${model_item}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}
    model_name=${model_item}_bs${batch_size}_${fp_item}

    log_with_profiler=$log_file
    profiler_path=$log_profile
    keyword="avg_batch_cost:"
    keyword_loss=""
    separator=""
    position=""
    range=""
    skip_steps=20
    model_mode=0
    ips_unit='sequences/s'
    index="1"
    gpu_num=$num_gpu_devices
}


function _train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"

    profiler_cmd=""
    profiler_options="batch_range=[100,110];profile_path=${log_profile}"
    if [ $need_profile = "on" ]; then
        profiler_cmd="--profiler_options=${profiler_options}"
    fi

    train_cmd="${profiler_cmd}
               --model_name_or_path=${model_item}
               --task_name=SST-2
               --max_seq_length=128
               --pad_to_max_seq_len=True
               --logging_steps=1
               --save_steps=2000
               --batch_size=${batch_size}
               --learning_rate=2e-5
               --max_steps=${max_iter}
               --output_dir=${run_log_path}
    "

    case ${run_mode} in
    sp)
        train_cmd="python -m paddle.distributed.launch --gpus=$CUDA_VISIBLE_DEVICES \
        examples/language_model/xlnet/run_glue.py ${train_cmd}" ;;
    mp)
        train_cmd="python -m paddle.distributed.launch --gpus=$CUDA_VISIBLE_DEVICES \
        examples/language_model/xlnet/run_glue.py ${train_cmd}"
        log_parse_file="mylog/workerlog.0" ;;
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
    #kill -9 `ps -ef|grep 'python'|awk '{print $2}'`

    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
#_train
_run
