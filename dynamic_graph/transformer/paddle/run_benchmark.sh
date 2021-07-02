#!bin/bash

set -xe
if [[ $# -lt 1 ]]; then
    echo "running job dict is {1: speed, 2:mem, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1|2|3 sp|mp 100(max_iter) base|big(model_type) fp32|amp_fp16(fp_mode)"
    exit
fi

function _set_params(){
    index=$1
    base_batch_size=4096

    run_mode=${2}
    max_iter=${3}
    model_name="transformer_"${4}
    fp_mode=${5:-"fp32"}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi
 
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}

    mission_name="机器翻译"
    direction_id=1
    skip_steps=3
    keyword="ips:"
    model_mode=-1
    ips_unit="words/s"

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    log_file=${run_log_path}/dynamic_${model_name}_bs${base_batch_size}_${fp_mode}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/dynamic_${model_name}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_dynamic_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}
}

function _train(){
    if [ ${model_name} == "transformer_base" ]; then 
        config_file="transformer.base.yaml"
    elif [ ${model_name} == "transformer_big" ]; then
        config_file="transformer.big.yaml"
    else
        echo " The model should be transformer_big or transformer_base!"
        exit 1
    fi

    # 混合精度监控。不支持传参修改。fp16 和fp32 混合，无论哪种情况需设置对应值，防止参数错误
    if [ ${fp_mode} == "amp_fp16" ]; then
        sed -i "s/^use_amp.*/use_amp: True/g" ./configs/${config_file}
        sed -i "s/^use_pure_fp16.*/use_pure_fp16: False/g" ./configs/${config_file}
    elif [ ${fp_mode} == "fp32" ]; then
        sed -i "s/^use_amp.*/use_amp: False/g" ./configs/${config_file}
        sed -i "s/^use_pure_fp16.*/use_pure_fp16: False/g" ./configs/${config_file}
    else
        echo " The fp_mode should be fp32 or amp_fp16"
    fi

    sed -i "s/^max_iter.*/max_iter: ${max_iter}/g" ./configs/${config_file} #不支持传参修改
    model_name=${model_name}_bs${base_batch_size}_${fp_mode}

    train_cmd="--config ./configs/${config_file} --benchmark"

    if [ ${run_mode} = "sp" ]; then
        train_cmd="python -u train.py "${train_cmd}
    else
        rm -rf ./mylog
        train_cmd="python -m paddle.distributed.launch  --gpus=$CUDA_VISIBLE_DEVICES  --log_dir ./mylog train.py "${train_cmd}
        log_parse_file="mylog/workerlog.0"
    fi

    timeout 15m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
        export job_fail_flag=1
    else
        echo -e "${model_name}, SUCCESS"
        export job_fail_flag=0
    fi
    if [ ${run_mode} != "sp"  -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_run
