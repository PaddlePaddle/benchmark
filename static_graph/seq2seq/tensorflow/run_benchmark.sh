#!/bin/bash

set -x

if [[ $# -lt 1 ]]; then
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh speed|mem|maxbs sp|mp /ssd1/ljh/logs"
    exit
fi

function _set_params() {
    index=$1
    base_batch_size=128
    model_name=seq2seq

    run_mode="sp" # Don't support multi-process running.
    run_log_root=${3:-$(pwd)}

    skip_steps=10
    keyword="step-time"
    separator=" "
    position=5
    range=-1
    model_mode=2  # s/step -> steps/s

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}
    if [ ${num_gpu_devices} -gt 1 ]; then
        echo "Multi-GPU training is not supported yet."
        exit
    fi

    log_file=${run_log_root}/${model_name}_${index}_${num_gpu_devices}_${run_mode}


}

function _set_env(){
    echo "nothing"
}

function _train() {
    if [ $index = "speed" ]; then
        sed -i '145c \  config_proto.gpu_options.allow_growth = False' nmt/nmt/utils/misc_utils.py
    elif [ $index = "mem" ]; then
        echo "this index is: "$index
        sed -i '145c \  config_proto.gpu_options.allow_growth = True' nmt/nmt/utils/misc_utils.py
    fi

    WORK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")" && pwd )"
    data_path=${WORK_ROOT}/nmt/data/en-vi
    cd $WORK_ROOT/nmt
    rm -rf ./outputs/*
    python -m nmt.nmt \
        --attention=scaled_luong \
        --optimizer adam \
        --learning_rate 0.001 \
        --src=en --tgt=vi \
        --vocab_prefix=${data_path}/vocab  \
        --train_prefix=${data_path}/train \
        --dev_prefix=${data_path}/tst2012  \
        --test_prefix=${data_path}/tst2013 \
        --out_dir=./outputs \
        --batch_size=${base_batch_size} \
        --num_train_steps=12000 \
        --steps_per_stats=1 \
        --num_layers=2 \
        --num_units=512 \
        --dropout=0.2 \
        --metrics=bleu \
        --num_buckets=5 > ${log_file} 2>&1 &
     seq2seq_train_pid=$!
     sleep 300
     kill $seq2seq_train_pid
}

source ${BENCHMARK_ROOT}/competitive_products/common_scripts/run_model.sh
_set_params $@
_set_env
_run
