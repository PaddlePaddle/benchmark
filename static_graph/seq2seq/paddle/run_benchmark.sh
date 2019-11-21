#!bin/bash
set -xe

if [[ $# -lt 1 ]]; then
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh speed|mem|maxbs sp|mp 1(max_epoch) 1|0(is_profiler)"
    exit
fi

function _set_params(){
    index=$1
    base_batch_size=128
    model_name="seq2seq"

    run_mode="sp" # Don't support mp
    max_epoch=${3}
    is_profiler=${4:-0}
 
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}

    skip_steps=0
    keyword="avg_time:"
    separator=" "
    position=-2
    model_mode=2 # s/step -> steps/s

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}
    if [ ${num_gpu_devices} -gt 1 ]; then
        echo "Multi-GPU training is not supported yet."
        exit
    fi

    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}
}

function _set_env(){
    #开启gc
    echo "nothing"
}

function _train(){
   train_cmd="--src_lang en --tar_lang vi \
              --attention True \
              --num_layers 2 \
              --hidden_size 512 \
              --src_vocab_size 17191 \
              --tar_vocab_size 7709 \
              --batch_size ${base_batch_size} \
              --dropout 0.2 \
              --init_scale  0.1 \
              --max_grad_norm 5.0 \
              --train_data_prefix data/en-vi/train \
              --eval_data_prefix data/en-vi/tst2012 \
              --test_data_prefix data/en-vi/tst2013 \
              --vocab_prefix data/en-vi/vocab \
              --use_gpu True \
              --profiler_path=${profiler_path} \
              --max_epoch=${max_epoch}"
    
    if [[ ${is_profiler} -eq 1 ]]; then
        python -u train.py \
               --profile \
               ${train_cmd} > ${log_file} 2>&1
    elif [[ ${is_profiler} -eq 0 ]]; then
        python -u train.py ${train_cmd} > ${log_file} 2>&1
    fi
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_set_env
_run
