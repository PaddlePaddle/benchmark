#!/usr/bin/env bash
set -xe

# 参数说明
function _set_params(){
    run_mode=${1:-"sp"}
    batch_size=${2:-"64"}
    fp_item=${3:-"fp32"}        # fp32|fp16
    max_iter=${4}       # 如果需要修改代码提前中断
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}

    model_name="nlp_modelName"
    mission_name="语义表示"            # 模型所属任务名称，具体可参考scripts/config.ini  必填）
    direction_id=1                   # 任务所属方向，0：CV，1：NLP，2：Rec。     (必填)
    ips_unit="sequences/s"

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    log_file=${run_log_path}/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}
    index_log_file=${run_log_path}/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}_speed
}

function _train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
    # 防止checkpoints冲突
    rm -rf results/checkpoints
    # 如需开启特殊优化flag、参数请注明

    train_cmd="--model_name_or_path=${model_name}
               --logging_dir=${run_log_path}
               --task_name=sst2
               --max_seq_length=128
               --per_device_train_batch_size=${batch_size}
               --learning_rate=2e-5
               --num_train_epochs=1
               --max_steps=1000
               --pad_to_max_length=True
               --logging_strategy=steps
               --logging_steps=500
               --do_train
               "

    case ${run_mode} in
    sp)
        train_cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python examples/pytorch/text-classification/run_glue.py ${train_cmd}" ;;
    mp)
        train_cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python examples/pytorch/text-classification/run_glue.py ${train_cmd}" ;;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac

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
}

function _analysis_log(){
    python analysis_log.py ${log_file} ${index_log_file}   # 分析log产出待入库的json 文件
}

_set_params $@
_train
_analysis_log
