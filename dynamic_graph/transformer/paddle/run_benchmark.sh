#!bin/bash

set -xe
if [[ $# -lt 1 ]]; then
    echo "running job dict is {1: speed, 2:mem, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1|2|3 sp|mp 100(max_iter)"
    exit
fi

function _set_params(){
    index=$1
    base_batch_size=4096
    model_name="transformer"

    run_mode=${2}
    max_iter=${3}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi
 
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}

    mission_name="机器翻译"
    direction_id=1
    skip_steps=0
    keyword="avg_speed:"
    separator=" "
    position=17
    model_mode=1 # s/step -> steps/s

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    log_file=${run_log_path}/dynamic_${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/dynamic_${model_name}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_dynamic_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}
}

function _train(){
   train_cmd="--max_iter ${max_iter} \
              --src_vocab_fpath gen_data/iwslt14.tokenized.de-en/vocab.de \
              --trg_vocab_fpath gen_data/iwslt14.tokenized.de-en/vocab.en \
              --special_token  <s> <e> <unk> \
              --training_file gen_data/iwslt14.tokenized.de-en/para_small.de-en \
              --validation_file gen_data/wmt16_ende_data_bpe/newstest2014.tok.bpe.32000.en-de \
              --predict_file gen_data/wmt16_ende_data_bpe/newstest2016.tok.bpe.32000.en-de \
              --weight_sharing False \
              --batch_size ${base_batch_size}"

    if [ ${run_mode} = "sp" ]; then
        train_cmd="python -u train.py "${train_cmd}
    else
        rm -rf ./mylog
        train_cmd="python -m paddle.distributed.launch --started_port 9798 --selected_gpus=$CUDA_VISIBLE_DEVICES  --log_dir ./mylog train.py "${train_cmd}
        log_parse_file="mylog/workerlog.0"
    fi

    ${train_cmd} > ${log_file} 2>&1
    if [ ${run_mode} != "sp"  -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_run
