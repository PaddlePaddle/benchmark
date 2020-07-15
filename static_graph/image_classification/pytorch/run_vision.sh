#!/bin/bash

set -x

if [ $# -lt 3 ]; then
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_image_resnet.sh speed|mem|maxbs 32 resnet50|resnet101 sp|mp /ssd1/ljh/logs"
    exit
fi

function _set_params() {
    index=$1
    base_batch_size=$2
    model_name=$3 # resnet50, resnet101
    run_mode="sp"
    run_log_root=${5:-$(pwd)}

#    skip_steps=2
#    keyword="img/s:"
#    separator=" "
#    position=14
#    model_mode=1 # s/step -> samples/s
    mission_name="图像分类"           # 模型所属任务名称，具体可参考scripts/config.ini                                （必填）
    direction_id=0                   # 任务所属方向，0：CV，1：NLP，2：Rec。                                         (必填)
    skip_steps=2
    keyword="time:"
    separator=" "
    position=29
    model_mode=0 # s/step -> samples/s

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    batch_size=`expr $base_batch_size \* $num_gpu_devices`
    num_workers=`expr 8 \* $num_gpu_devices`

    if [[ ${index} = "analysis" ]]; then
        log_file=${run_log_root}/log_vision_${model_name}_speed_${num_gpu_devices}_${run_mode}
    else
        log_file=${run_log_root}/log_vision_${model_name}_${index}_${num_gpu_devices}_${run_mode}
    fi
    log_parse_file=${log_file}
}

function _train() {
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
    WORK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")" && pwd )"
#    data_path=/ssd1/ljh/dataset/pytorch_image_classification
    data_path="/ssd2/liyang/benchmark/benchmark/static_graph/image_classification/pytorch/ILSVRC2012/" 
    num_epochs=2

    python -c "import torch; print(torch.__version__)"
    export PYTHONPATH=${BENCHMARK_ROOT}/third_party/pytorch/vision

    echo "${model_name}, batch_size: ${batch_size}"
    stdbuf -oL python ${BENCHMARK_ROOT}/third_party/pytorch/vision/references/classification/train.py \
           --data-path ${data_path} \
           --model ${model_name} \
           --device cuda \
           --batch-size ${batch_size} \
           --epochs ${num_epochs} \
           --workers ${num_workers} \
           --print-freq 10 \
           --output-dir ./output/vision \
           --cache-dataset > ${log_file} 2>&1 &

    train_pid=$!
    sleep 500
    kill -9 `ps -ef|grep python |awk '{print $2}'`
}

source ${BENCHMARK_ROOT}/competitive_products/common_scripts/run_model.sh
_set_params $@
_run
