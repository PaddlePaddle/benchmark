#!bin/bash
set -x

if [[ $# -lt 4 ]]; then
    echo "running job dict is {1: speed, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 1|3|6 128 sp|mp 1(max_epoch) mode"
    exit
fi

function _set_params(){
    index=$1                         # 速度(speed)|显存占用(mem)|单卡最大支持batch_size(maxbs)                       (必填)
    base_batch_size=$2               # 单卡的batch_size，如果固定的，可以写死。                                       (必填
）
    mode=${5}
    model_name="ResNet50_bs${base_batch_size}_${mode}_fp16" # 模型名字如："SE-ResNeXt50"，如果是固定的，可以写死，如果需要其他参数可以参考bert实现（必填）
    run_mode=${3:-"sp"}              # 单进程(sp)|多进程(mp)，默认单进程                                            （必填）
    mission_name="图像分类"          # 模型所属任务名称，具体可参考scripts/config.ini                                （必填）
    direction_id=0                   # 任务所属方向，0：CV，1：NLP，2：Rec。                                         (必填)

    max_epoch=${4}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi

    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}

    skip_steps=8                     # 解析日志，有些模型前几个step耗时长，需要跳过                                    (必填)
    keyword="INFO: epoch"                 # 解析日志，筛选出数据所在行的关键字                                             (必填)
    separator=": "                    # 解析日志，数据所在行的分隔符                                                  (必填)
    position=6                      # 解析日志，按照分隔符分割后形成的数组索引                                        (必填)
    model_mode=0                     # 解析日志，具体参考scripts/analysis.py.                                      (必填)
    range=0:6
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
    export FLAGS_conv_workspace_size_limit=4000 #MB
    export FLAGS_cudnn_exhaustive_search=1
    export FLAGS_cudnn_batchnorm_spatial_persistent=1
}

function _train(){
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"

    export CUDA_VISIBLE_DEVICES=0

    if [ ${mode} == "amp" ]; then
        USE_DALI=True
        USE_PURE_FP16=False
        USE_AMP=True
        MULTI_PRECISION=${USE_PURE_FP16}
    elif [ ${mode} == "pure" ]; then
        USE_DALI=True
        USE_PURE_FP16=True
        USE_AMP=False
        MULTI_PRECISION=${USE_PURE_FP16}
    else
        echo "check your mode!"
    fi

    if [ ${USE_DALI} == "True" ]; then
        export FLAGS_fraction_of_gpu_memory_to_use=0.8
    fi
    
    train_cmd="-c ./configs/ResNet/ResNet50_fp16.yml
            -o TRAIN.batch_size=${batch_size}
            -o validate=False
            -o epochs=${max_epoch}
            -o TRAIN.data_dir=./dataset/imagenet100_data
            -o TRAIN.file_list=./dataset/imagenet100_data/train_list.txt
            -o TRAIN.num_workers=8
            -o print_interval=10
            -o data_format=NHWC
            -o use_dali=${USE_DALI}
            -o use_amp=${USE_AMP}
            -o use_pure_fp16=${USE_PURE_FP16}
            -o multi_precision=${MULTI_PRECISION}
            -o use_gpu=True
            -o image_shape=[4,224,224]
            "
            #-o is_distributed=False \
            #-o image_shape "[4, 224, 224]"
    

    case ${run_mode} in
    sp) train_cmd="python -m paddle.distributed.launch tools/static/train.py "${train_cmd} ;;
    mp)
        rm -rf ./mylog
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --gpus=$CUDA_VISIBLE_DEVICES tools/static/train.py "${train_cmd}
        log_parse_file="mylog/workerlog.0" ;;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac

    timeout 15m ${train_cmd} > ${log_file} 2>&1 
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
