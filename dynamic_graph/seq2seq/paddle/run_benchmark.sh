#!bin/bash

set -xe
if [[ $# -lt 1 ]]; then
    echo "running job dict is {1: speed, 2:mem, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 1|2|3 1(max_epoch)"
    exit
fi

function _set_params(){
    index=$1
    base_batch_size=128
    model_name="seq2seq"_bs${base_batch_size}

    run_mode="sp"
    max_epoch=${2}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi
 
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}
    
    direction_id=1
    mission_name="文本生成"
    skip_steps=0
    keyword="ips:"
    model_mode=-1
    ips_unit="tokens/s"

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
    train_cmd_en_vi="--src_lang en --tar_lang vi \
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
                     --max_epoch ${max_epoch} \
                     --model_path attention_models"
    train_cmd_de_en="--src_lang de --tar_lang en \
                     --attention True \
                     --num_layers 4 \
                     --hidden_size 1000 \
                     --src_vocab_size 8847 \
                     --tar_vocab_size 6631 \
                     --batch_size ${base_batch_size} \
                     --dropout 0.2 \
                     --init_scale  0.1 \
                     --max_grad_norm 5.0 \
                     --train_data_prefix data/iwslt14.tokenized.de-en/train \
                     --eval_data_prefix data/iwslt14.tokenized.de-en/valid \
                     --test_data_prefix data/iwslt14.tokenized.de-en/test \
                     --vocab_prefix data/iwslt14.tokenized.de-en/vocab \
                     --use_gpu True \
                     --max_epoch ${max_epoch} \
                     --model_path attention_models"

    sed -i '/dev_ppl = eval(valid_data)/d' train.py
    sed -i '/print("dev ppl", dev_ppl)/d' train.py
    sed -i '/test_ppl = eval(test_data)/d' train.py
    sed -i '/print("test ppl", test_ppl)/d' train.py

    timeout 15m python -u train.py ${train_cmd_de_en} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
        export job_fail_flag=1
    else
        echo -e "${model_name}, SUCCESS"
        export job_fail_flag=0
    fi
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_run
