#!bin/bash
set -xe

if [[ $# -lt 4 ]]; then
    echo "running job dict is {1: speed, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 1|3|6 32 nextvlad|CTCN sp|mp 2(max_epoch)"
    exit
fi

function _set_params(){
    index=$1                         # 速度(speed)|显存占用(mem)|单卡最大支持batch_size(maxbs)                        （必填）
    base_batch_size=$2               # 单卡的batch_size，如果固定的，可以写死                                         （必填）
    model_name=${3}_bs${base_batch_size}                    # 模型名字如："SE-ResNeXt50"，如果是固定的，可以写死，如果需要其他参数可以参考bert实现（必填）
    run_mode=${4:-"sp"}              # 单进程(sp)|多进程(mp)，默认单进程                                              （必填）
    max_epoch=${5}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi
 
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}

    mission_name="视频分类"           # 模型所属任务名称，具体可参考scripts/config.ini                                （必填）
    direction_id=0                    # 任务所属方向，0：CV，1：NLP，2：Rec。                                         （必填）
    keyword="ips:"             # 解析日志，筛选出数据所在行的关键字                                            （必填）
    skip_steps=1                      # 解析日志，有些模型前几个step耗时长，需要跳过                                  （必>填）
    model_mode=-1 # s/step -> samples/s
    ips_unit="images/s"

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    if [ $run_mode = "sp" ]; then
        batch_size=`expr $base_batch_size \* $num_gpu_devices`
    else
        batch_size=$base_batch_size
    fi

    config_file_name="nextvlad.yaml"

    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/${model_name}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}

}

function _set_env(){
    #开启gc
    export FLAGS_eager_delete_tensor_gb=0.0
    export FLAGS_fraction_of_gpu_memory_to_use=0.98
}

function _train(){
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
    WORK_ROOT=$PWD
    echo "${model_name}, batch_size: ${batch_size}"
    sed -i "s/num_gpus: [1-8]/num_gpus: ${num_gpu_devices}/g" ./configs/${config_file_name}

    train_cmd=" --model_name ${model_name%_bs*} \
        --config ./configs/${config_file_name} \
        --valid_interval 1 \
        --log_interval 10 \
        --batch_size=$batch_size \
        --is_profiler=${is_profiler} \
        --profiler_path=${profiler_path} \
        --epoch=${max_epoch}"

    case ${run_mode} in
    sp) train_cmd="python -u train.py "${train_cmd} ;;
    mp)
        rm -rf ./mylog
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --gpus=$CUDA_VISIBLE_DEVICES train.py "${train_cmd}
        log_parse_file="mylog/workerlog.0" ;;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac

    timeout 15m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
        export job_fail_flag=1
    else
        echo -e "${model_name}, SUCCESS"
        export job_fail_flag=0
    fi
    kill -9 `ps -ef|grep python |awk '{print $2}'`

    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
    cd ${WORK_ROOT}
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_set_env
_run
