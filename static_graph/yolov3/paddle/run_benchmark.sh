#!bin/bash
set -xe

if [[ $# -lt 1 ]]; then
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh speed|mem|maxbs sp|mp 1|0(profile switch) /ssd1/ljh/logs profiler_dir"
    exit
fi

function _set_params(){
    index="$1"
    run_mode=${2:-"sp"}
    is_profiler=${3}
    run_log_path=${4:-$(pwd)}
    profiler_dir=${5:-$(pwd)}

    model_name="yolov3"
    skip_steps=5
    keyword="Iter"
    separator=" "
    position=-1
    model_mode=0

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}

    if [[ ${index} = "maxbs" ]]; then base_batch_size=14; else base_batch_size=8; fi
    batch_size=`expr ${base_batch_size} \* ${num_gpu_devices}`

    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_parse_file=${log_file}

}

function _set_env(){
    export FLAGS_eager_delete_tensor_gb=0.0
    export FLAGS_fraction_of_gpu_memory_to_use=0.98
    export FLAGS_memory_fraction_of_eager_deletion=1.0
}

function _train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
    if [[ ${run_mode} == "sp" && ${num_gpu_devices} -eq 8 ]]; then
        num_workers=32
    else
        num_workers=8
    fi

    train_cmd=" --model_save_dir=output/ \
     --pretrain=./weights/darknet53/ \
     --data_dir=./dataset/coco/ \
     --batch_size=${base_batch_size} \
     --syncbn=True \
     --worker_num=${num_workers} \
     --is_profiler=${is_profiler} \
     --profiler_path=${profiler_dir}/profiler_${model_name}"

    case ${run_mode} in
    sp) train_cmd="python -u train.py "${train_cmd} ;;
    mp)
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --selected_gpus=$CUDA_VISIBLE_DEVICES train.py "${train_cmd}
        log_parse_file="mylog/workerlog.0" ;;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac

    ${train_cmd} > ${log_file} 2>&1 &
    train_pid=$!
#    sleep 600
##############
    total_sleep=0
    while [ $total_sleep -le 600 ] && [ `ps -ax | awk '{print $1}' | grep -e "^${train_pid}$"` ] #whether the pid of train.py exists
    do
      sleep 5
      let total_sleep=total_sleep+5 
    done
###############
    kill -9 `ps -ef|grep 'python'|awk '{print $2}'`

    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_set_env
_run
