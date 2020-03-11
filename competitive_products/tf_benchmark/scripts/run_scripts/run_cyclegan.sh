#!bin/bash
set -x

if [[ $# -lt 1 ]]; then
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_cyclegan.sh speed|mem  /log/path"
    exit
fi

function _set_params(){
    index="$1"
    run_log_path=${2:-$(pwd)}

    model_name="CycleGAN"
    skip_steps=3
    keyword="Time cost:"
    separator=" "
    position=-1
    model_mode=0
    run_mode=sp

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}
    base_batch_size=1
    log_file=${run_log_path}/log_${model_name}_${index}_${num_gpu_devices}
}

function _set_env(){
    echo "nothing"
}


function _train(){
    if [ $index = "mem" ];then
         sed -i '56c TEST_GPU_MEM = True' main.py
    elif [ $index = "speed" ];then
         sed -i '56c TEST_GPU_MEM = False' main.py
    fi

    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=${base_batch_size}"
    python main.py > ${log_file} 2>&1 &
    train_pid=$!
    sleep 300
    kill -9 $train_pid
}

source ${TF_BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_set_env
_run
