#!bin/bash

set -xe
if [[ $# -lt 1 ]]; then
    echo "running job dict is {1: speed, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 1|3 sp|mp"
    exit
fi

function _set_params(){
    index=$1
    base_batch_size=128
    model_name="seq2seq"

    run_mode=${2:-"sp"}
    max_iter=${3}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi
 
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}

    direction_id=1
    mission_name="文本生成"
    skip_steps=1
    keyword="avg_speed:"
    separator=" "
    position=4
    model_mode=1 # steps/s -> steps/s

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    log_file=${run_log_path}/dynamic_${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_parse_file=${log_file}
}

function _train(){
    fairseq-train data-bin/iwslt14.tokenized.de-en \
              --arch lstm_luong_wmt_en_de  \
              --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
              --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
              --dropout 0.3 --weight-decay 0.0001 \
              --criterion cross_entropy \
              --log-format 'simple' --log-interval 100 \
              --use-basic \
              --max-sentences ${base_batch_size} \
              --decoder-attention False > ${log_file} 2>&1 &

    train_pid=$!
    sleep 1200
    kill -9 `ps -ef|grep python |awk '{print $2}'`
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_run
