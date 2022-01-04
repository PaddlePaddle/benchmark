#!bin/bash
set -xe

if [[ $# -lt 3 ]]; then
    echo "running job dict is {1: speed, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1|3 large|medium|small static|padding sp|mp 3(max_epoch)"
    exit
fi

function _set_params(){
    index=$1
    model_type=$2
    rnn_type=$3
    run_mode=${4:-"sp"}
    max_epoch=${5}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi

    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}

    if [[ ${index} -eq 6 ]]; then base_batch_size=12000; else base_batch_size=20; fi
    model_name="paddingrnn_"${model_type}_${rnn_type}_bs${base_batch_size}
    mission_name="语言模型"           # 模型所属任务名称，具体可参考scripts/config.ini                                （必填）
    direction_id=1                   # 任务所属方向，0：CV，1：NLP，2：Rec。                                         (必填)
    skip_steps=1                     # 解析日志，有些模型前几个step耗时长，需要跳过                                    (必填)
    keyword="avg_time:"              # 解析日志，筛选出数据所在行的关键字                                             (必填)
    separator=" "                    # 解析日志，数据所在行的分隔符                                                  (必填)
    position=8                       # 解析日志，按照分隔符分割后形成的数组索引                                        (必填)
    model_mode=1                     # 解析日志，具体参考scripts/analysis.py.                                      (必填)

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    batch_size=`expr ${base_batch_size} \* $num_gpu_devices`
    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/${model_name}_3_${num_gpu_devices}_${run_mode}
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
    python -c "import paddle; print(paddle.__version__)"
    train_cmd="--data_path data/simple-examples/data/ \
      --model_type $model_type \
      --use_gpu True \
      --enable_ce \
      --max_epoch=${max_epoch} \
      --rnn_model ${rnn_type} \
      --use_dataloader True \
      --enable_auto_fusion True \
      --profile ${is_profiler} \
      --profiler_path=${profiler_path} \
      --batch_size ${batch_size}"
     timeout 15m python -u train.py ${train_cmd} > ${log_file} 2>&1
     if [ $? -ne 0 ];then
         echo -e "${model_name}, FAIL"
         export job_fail_flag=1
     else
         echo -e "${model_name}, SUCCESS"
         export job_fail_flag=0
     fi
     kill -9 `ps -ef|grep python |awk '{print $2}'`
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_set_env
_run
