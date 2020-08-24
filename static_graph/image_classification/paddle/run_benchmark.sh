#!bin/bash
set -xe

if [[ $# -lt 4 ]]; then
    echo "running job dict is {1: speed, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1|3|6 32 model_name sp|mp 1000(max_iter)"
    exit
fi

function _set_params(){
    index=$1                         # 速度(speed)|显存占用(mem)|单卡最大支持batch_size(maxbs)                       (必填)
    base_batch_size=$2               # 单卡的batch_size，如果固定的，可以写死。                                       (必填）
    model_name=$3                    # 模型名字如："SE-ResNeXt50"，如果是固定的，可以写死，如果需要其他参数可以参考bert实现（必填）
    run_mode=${4:-"sp"}              # 单进程(sp)|多进程(mp)，默认单进程                                            （必填）
    mission_name="图像分类"           # 模型所属任务名称，具体可参考scripts/config.ini                                （必填）
    direction_id=0                   # 任务所属方向，0：CV，1：NLP，2：Rec。                                         (必填)

    max_iter=${5}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi

    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}

    skip_steps=8                     # 解析日志，有些模型前几个step耗时长，需要跳过                                    (必填)
    if [ ${run_mode} = "mp" ]; then skip_steps=31; fi
    keyword="elapse"                 # 解析日志，筛选出数据所在行的关键字                                             (必填)
    separator=" "                    # 解析日志，数据所在行的分隔符                                                  (必填)
    position=-2                      # 解析日志，按照分隔符分割后形成的数组索引                                        (必填)
    model_mode=0                     # 解析日志，具体参考scripts/analysis.py.                                      (必填)
    #range=-1                        # 解析日志，取得列表索引的值后，切片[0：range], 默认最后一位可以不用填, 或者 3:10格式

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    batch_size=`expr ${base_batch_size} \* ${num_gpu_devices}`
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
    if [[ ${model_name} == "ResNet50" && ${num_gpu_devices} == 1 ]]; then
        export FLAGS_cudnn_exhaustive_search=1
    fi
}

function _train(){
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
    WORK_ROOT=$PWD
    num_epochs=2
    echo "${model_name}, batch_size: ${batch_size}"
    if [ ${model_name} = "ResNet50_bs32" ] || [ ${model_name} = "ResNet101" ] || [ ${model_name} = "ResNet50_bs128" ];
    then
       train_cmd="--batch_size=${batch_size} \
           --total_images=1281167 \
           --class_dim=1000 \
           --model_save_dir=output/ \
           --lr_strategy=piecewise_decay \
           --num_epochs=${num_epochs} \
           --lr=0.1 \
           --max_iter=${max_iter} \
           --validate=0 \
           --is_profiler=${is_profiler} \
           --profiler_path=${profiler_path} \
           --l2_decay=1e-4"
    elif [ ${model_name} = "SE_ResNeXt50_32x4d" ];
    then
        train_cmd="--batch_size=${batch_size} \
           --total_images=1281167 \
           --class_dim=1000 \
           --model_save_dir=output/ \
           --data_dir=data/ILSVRC2012 \
           --lr_strategy=cosine_decay \
           --lr=0.1 \
           --l2_decay=1.2e-4 \
           --max_iter=${max_iter} \
           --is_profiler=${is_profiler} \
           --profiler_path=${profiler_path} \
           --validate=0 \
           --num_epochs=${num_epochs}"
    else
        echo "model: $model_name not support!"
	exit
    fi

    if [ ${model_name} = "SE_ResNeXt50_32x4d" ] || [ ${model_name} = "ResNet101" ]; then
        train_cmd="--model=${model_name} "${train_cmd}
    else
        train_cmd="--model=ResNet50 "${train_cmd}  # 必须这么写， 因模型有个内置的支持的model列表
    fi

    case ${run_mode} in
    sp) train_cmd="python -u train.py "${train_cmd} ;;
    mp)
        rm -rf ./mylog
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
    cd ${WORK_ROOT}
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_set_env
_run
