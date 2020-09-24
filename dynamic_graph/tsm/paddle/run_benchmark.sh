#!bin/bash
set -xe
if [[ $# -lt 1 ]]; then
    echo "running job dict is {1: speed, 2:mem, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 1|2|3 batch_size TSM sp|mp 1(max_epoch)"
    exit
fi

function _set_params(){
    index=$1
    base_batch_size=${2}
    model_name=${3}

    run_mode=${4:-"sp"} # Use sp for single GPU and mp for multiple GPU.
    max_epoch=${5:-"1"}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi
 
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}

    mission_name="视频分类"
    direction_id=0
    skip_steps=5
    keyword="batch_cost:"
    separator=" "
    position=-5
    model_mode=0 # s/step -> samples/s

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    log_file=${run_log_path}/dynamic_${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/dynamic_${model_name}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_dynamic_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}
    batch_size=`expr ${base_batch_size} \* ${num_gpu_devices}`
}

function _train(){
    if [ ${run_mode} == "sp" ]; then
        config_files="./tsm_ucf101_sing.yaml"
    elif [ ${run_mode} == "mp" ]; then
        config_files="./tsm_ucf101.yaml"
        sed -i "s/learning_rate: 0.01/learning_rate: 0.02/g" ${config_files} # RD 暂未支持传LR
        sed -i "s/num_gpus: 4/num_gpus: 8/g" ${config_files}
    else
        echo "------not support"
        exit
    fi

    # 去掉test，修改data根目录，当前实现里没有开关可以关闭或者修改
    grep -q "video_model.eval()" ./train.py
    if [ $? -eq 1 ]; then
        echo "----------already addressed disable test after train"
    else
        sed -i "/video_model.eval()/d" ./train.py 
        sed -i "/val(epoch, video_model, valid_config, args)/d" ./train.py 
        sed -i "s/ucf101_root = \"\/ssd4\/chaj\/ucf101\/\"/ucf101_root = \"\/ssd1\/ljh\/dataset\/dygraph_data\/TSM\/ucf101\/\"/g" ucf101_reader.py
    fi

    train_cmd="--epoch ${max_epoch} \
               --batch_size=${batch_size} \
               --config=${config_files} \
               --pretrain=./ResNet50_pretrained \
               --weights=k400_wei/TSM.pdparams
               "
    if [ ${run_mode} = "sp" ]; then
        train_cmd="python -u train.py --use_data_parallel=False "${train_cmd}
    else
        rm -rf ./mylog
        train_cmd="python -m paddle.distributed.launch --started_port=38989 --log_dir ./mylog train.py --use_data_parallel=True "${train_cmd}
        log_parse_file="mylog/workerlog.0"
    fi
    
    ${train_cmd} > ${log_file} 2>&1
    if [ ${run_mode} != "sp"  -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_run
