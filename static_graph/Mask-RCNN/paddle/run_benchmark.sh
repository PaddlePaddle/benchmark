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

    model_name="mask_rcnn"
    skip_steps=3
    keyword="loss_rpn_bbox"
    separator=" "
    position=19
    model_mode=0

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    if [[ $index = "maxbs" ]]; then base_batch_size=5; else base_batch_size=1; fi
    batch_size=`expr ${base_batch_size} \* $num_gpu_devices`
    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}

}

function _set_env(){
   export FLAGS_eager_delete_tensor_gb=0.0
   export FLAGS_fraction_of_gpu_memory_to_use=0.98
   export FLAGS_memory_fraction_of_eager_deletion=1.0
   export FLAGS_conv_workspace_size_limit=500
}

function _train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"

    train_cmd=" --model_save_dir=output/ \
     --pretrained_model=./imagenet_resnet50_fusebn/ \
     --data_dir=./dataset/coco \
     --im_per_batch=${base_batch_size} \
     --max_iter=${max_iter} \
     --is_profiler=${is_profiler} \
     --profiler_path=${profiler_path} \
     --MASK_ON=True"

    case ${run_mode} in
    sp) train_cmd="python -u train.py "${train_cmd} ;;
    mp)
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --selected_gpus=$CUDA_VISIBLE_DEVICES train.py "${train_cmd}
        log_parse_file="mylog/workerlog.0" ;;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac

    ${train_cmd} > ${log_file} 2>&1
    kill -9 `ps -ef|grep python |awk '{print $2}'`

    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_set_env
_run
