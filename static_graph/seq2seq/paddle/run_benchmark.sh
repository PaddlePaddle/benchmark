#!bin/bash
set -xe

if [[ $# -lt 1 ]]; then
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh speed|mem|maxbs sp|mp 1|0(profiler switch) /ssd1/ljh/logs profiler_dir"
    exit
fi

function _set_params(){
    index=$1
    base_batch_size=128
    model_name="seq2seq"

    run_mode="sp" # Don't support mp
    is_profiler=${3}
    run_log_root=${4:-$(pwd)}
    profiler_dir=${5:-$(pwd)}
    profiler_path=${profiler_dir}/profiler_${model_name}

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

    log_file=${run_log_root}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_parse_file=${log_file}
}

function _set_env(){
    #开启gc
    echo "nothing"
}

function _train(){
   echo "Train on ${num_gpu_devices} GPUs"
   echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, base_batch_size=$base_batch_size, is_profiler=${is_profiler}, profiler_file_path=${profiler_path}"
 
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
          --is_profiler ${is_profiler} \
          --profiler_path ${profiler_path} \
          --max_epoch 2  > ${log_file} 2>&1
#   train_pid=$!
#   total_sleep=0
#   while [ `ps -ax | awk '{print$1}' | grep -e "^${train_pid}$"` ]
#   do
#     sleep 5
#     #let sleep=sleep+5
#   done
#   kill -9 `ps -ef|grep python |awk '{print $2}'`
     
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_set_env
_run
