#!bin/bash
set -xe

if [[ $# -lt 1 ]]; then
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh speed|mem|maxbs sp|mp /ssd1/ljh/logs"
    exit
fi

function _set_params(){
    index=$1
    base_batch_size=128
    model_name="seq2seq"

    run_mode=${2:-"sp"}
    run_log_path=${3:-$(pwd)}

    skip_steps=0
    keyword="avg_time:"
    separator=" "
    position=-2
    model_mode=1

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_parse_file=${log_file}

}

function _set_env(){
    #开启gc
    echo "nothing"
}

function _train(){
   python train.py \
          --src_lang en --tar_lang vi \
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
          --max_epoch 2  > ${log_file} 2>&1
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_set_env
_run