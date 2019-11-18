#!bin/bash
set -xe

if [[ $# -lt 1 ]]; then
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh speed|mem|maxbs sp|mp 1000(max_iter) 1|0(is_profiler)"
    exit
fi

function _set_params(){
    index="$1"
    run_mode=${2:-"sp"}

    max_iter=${3}
    is_profiler=${4:-0}

    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}

    model_name="DeepLab_V3+"
    skip_steps=1
    keyword="step_time_cost:"
    separator=" "
    position=5
    model_mode=0

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    if [[ ${index} = "maxbs" ]]; then base_batch_size=9; else base_batch_size=2; fi
    batch_size=`expr ${base_batch_size} \* ${num_gpu_devices}`

    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi

    log_parse_file=${log_file}

}

function _set_env(){
   export FLAGS_eager_delete_tensor_gb=0.0
   export FLAGS_fast_eager_deletion_mode=1
}

function _train(){
    DATASET_PATH=${PWD}/data/cityscape/
    INIT_WEIGHTS_PATH=${PWD}/deeplabv3plus_xception65_initialize
    SAVE_WEIGHTS_PATH=${PWD}/output/model
    train_crop_size=513
#    total_step=240
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"

    train_cmd=" --batch_size=${batch_size} \
        --train_crop_size=${train_crop_size} \
        --total_step=${max_iter} \
        --init_weights_path=${INIT_WEIGHTS_PATH} \
        --save_weights_path=${SAVE_WEIGHTS_PATH} \
        --dataset_path=${DATASET_PATH} \
        --parallel=True \
        --is_profiler=${is_profiler} \
        --profiler_path=${profiler_path} \
        --use_multiprocessing=True "

    case ${run_mode} in
    sp) train_cmd="python -u train.py "${train_cmd} ;;
    mp)
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --selected_gpus=${CUDA_VISIBLE_DEVICES} train.py "${train_cmd}
        log_parse_file="mylog/workerlog.0" ;;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac

    ${train_cmd} > ${log_file} 2>&1
    # Python multi-processing is used to read images, so need to
    # kill those processes if the main train process is aborted.
    #ps -aux | grep "$PWD/train.py" | awk '{print $2}' | xargs kill -9
    kill -9 `ps -ef|grep 'deeplabv3+'|awk '{print $2}'`

    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_set_env
_run
