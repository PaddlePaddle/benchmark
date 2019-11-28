#!bin/bash
set -xe

if [[ $# -lt 3 ]]; then
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh speed|mem large|medium|small static|padding sp|mp 3(max_epoch) 1|0(profiler)"
    exit
fi

function _set_params(){
    index=$1
    model_type=$2
    rnn_type=$3
    run_mode=${4:-"sp"}
    #run_log_path=${5:-$(pwd)}
    max_epoch=${5}
    is_profiler=${6:-0}

    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}

    model_name="paddingrnn_"${model_type}_${rnn_type}
    skip_steps=0
    keyword="avg_time:"
    separator=" "
    position=8
    model_mode=1

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    if [[ ${index} = "maxbs" ]]; then base_batch_size=12000; else base_batch_size=20; fi
    batch_size=`expr ${base_batch_size} \* $num_gpu_devices`
    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}

}

function _set_env(){
    export MKL_NUM_THREADS=1
    export OMP_NUM_THREADS=1

    # Occupy all GPU memory (5% reserved actually)
    export FLAGS_fraction_of_gpu_memory_to_use=1.0
    export FLAGS_eager_delete_tensor_gb=0.0
    export FLAGS_memory_fraction_of_eager_deletion=0.5
    if [[ ${model_type} == "large" ]]; then
        export FLAGS_memory_fraction_of_eager_deletion=1.0
    fi
}

function _train(){
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
    train_cmd="--data_path data/simple-examples/data/ \
      --model_type $model_type \
      --use_gpu True \
      --enable_ce \
      --max_epoch=${max_epoch} \
      --rnn_model ${rnn_type} \
      --use_dataloader True \
      --profile ${is_profiler} \
      --profiler_path=${profiler_path} \
      --batch_size ${batch_size}"
     python -u train.py ${train_cmd} > ${log_file} 2>&1
     kill -9 `ps -ef|grep python |awk '{print $2}'`
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_set_env
_run
