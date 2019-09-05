#!bin/bash
set -xe

if [[ $# -lt 2 ]]; then
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh speed|mem|maxbs DCGAN|CGAN|Pix2pix sp|mp /ssd1/ljh/logs"
    exit
fi

function _set_params(){
    index="$1"
    model_name="$2"
    run_mode=${3:-"sp"}
    run_log_path=${4:-$(pwd)}

    skip_steps=5
    keyword="Batch_time_cost:"
    separator=":"
    position=-1
    model_mode=0

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    base_batch_size=0

    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_parse_file=${log_file}

}

function _set_env(){
    export FLAGS_cudnn_exhaustive_search=1
    export FLAGS_eager_delete_tensor_gb=0.0
}

function _train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices"
    if echo {DCGAN CGAN} | grep -w $model_name &>/dev/null
    then
        base_batch_size=32
        echo "${model_name}, batch_size: ${base_batch_size}"
        train_cmd=" --model_net $model_name \
           --dataset mnist   \
           --noise_size 100  \
           --batch_size ${base_batch_size}   \
           --epoch 10"
    elif echo {Pix2pix} | grep -w $model_name &>/dev/null
    then
        base_batch_size=1
        echo "${model_name}, batch_size: ${base_batch_size}"
        train_cmd=" --model_net $model_name \
           --dataset cityscapes     \
           --train_list data/cityscapes/pix2pix_train_list \
           --test_list data/cityscapes/pix2pix_test_list    \
           --crop_type Random \
           --dropout True \
           --gan_mode vanilla \
           --batch_size ${base_batch_size} \
           --epoch 200 \
           --crop_size 256 "
    else
        echo "model: $model_name not support!"
    fi

    train_cmd="python -u train.py "${train_cmd}

    ${train_cmd} > ${log_file} 2>&1 &
    train_pid=$!
    sleep 120
    kill -9 $train_pid
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_set_env
_run