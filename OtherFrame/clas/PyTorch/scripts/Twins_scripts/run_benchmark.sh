#!/usr/bin/env bash
set -xe
# 运行示例：CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} 500 ${model_name}
# 参数说明
function _set_params(){
    run_mode=${1:-"sp"}          # 单卡sp|多卡mp
    batch_size=${2:-"64"}
    fp_item=${3:-"fp32"}        # fp32|fp16
    epoch=${4:-"300"}       # 可选，如果需要修改代码提前中断
    model_name=${5:-"model_name"}
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
 
    train_cmd="--model ${model_name} --batch-size ${batch_size} --data-path data/imagenet --dist-eval --drop-path 0.3 --epochs ${epoch}"
    case ${run_mode} in
    sp) train_cmd="python main.py ${train_cmd}" ;;
    mp)
        train_cmd="python -m torch.distributed.launch --nproc_per_node=${num_gpu_devices} --use_env main.py ${train_cmd}" ;;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac
# 以下不用修改
    timeout 5m ${train_cmd} 2>&1|tee ${log_file}
    python analysis_log.py -f ${log_file} -m ${model_name} -b ${batch_size} -n ${num_gpu_devices}
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
        export job_fail_flag=1
    else
        echo -e "${model_name}, SUCCESS"
        export job_fail_flag=0
    fi
    kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
 
}
 
_set_params $@
_train
