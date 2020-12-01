#!bin/bash
set -xe

if [[ $# -lt 4 ]]; then
    echo "running job dict is {1: speed, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1|3|6 128 sp|mp 600(max_iter)"
    exit
fi

function _set_params(){
    index=$1                         # 速度(speed)|显存占用(mem)|单卡最大支持batch_size(maxbs)                       (必填)
    base_batch_size=$2               # 单卡的batch_size，如果固定的，可以写死。                                       (必填）
    model_name="ResNet50_bs${base_batch_size}_fp16" # 模型名字如："SE-ResNeXt50"，如果是固定的，可以写死，如果需要其他参数可以参考bert实现（必填）
    run_mode=${3:-"sp"}              # 单进程(sp)|多进程(mp)，默认单进程                                            （必填）
    mission_name="图像分类"          # 模型所属任务名称，具体可参考scripts/config.ini                                （必填）
    direction_id=0                   # 任务所属方向，0：CV，1：NLP，2：Rec。                                         (必填)

    max_iter=${4}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi

    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}

    skip_steps=8                     # 解析日志，有些模型前几个step耗时长，需要跳过                                    (必填)
    if [ ${run_mode} = "mp" ]; then skip_steps=31; fi
    keyword="batch_cost"                 # 解析日志，筛选出数据所在行的关键字                                             (必填)
    separator=" "                    # 解析日志，数据所在行的分隔符                                                  (必填)
    position=14                      # 解析日志，按照分隔符分割后形成的数组索引                                        (必填)
    model_mode=0                     # 解析日志，具体参考scripts/analysis.py.                                      (必填)

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
    
    DATA_FORMAT="NHWC"
    USE_FP16=true #whether to use float16
    USE_DALI=true
    USE_ADDTO=true
    if ${USE_ADDTO} ;then
        export FLAGS_max_inplace_grad_add=8
    fi

    if ${USE_DALI}; then
        export FLAGS_fraction_of_gpu_memory_to_use=0.8
    fi
    num_epochs=2
    
    echo "${model_name}, batch_size: ${batch_size}"
    train_cmd="--model=ResNet50 \
               --data_dir=./data/ILSVRC2012/ \
               --batch_size=${batch_size} \
               --total_images=1281167 \
               --image_shape 4 224 224 \
               --class_dim=1000 \
               --print_step=10 \
               --model_save_dir=output/ \
               --lr_strategy=piecewise_decay \
               --use_fp16=${USE_FP16} \
               --scale_loss=128.0 \
               --use_dynamic_loss_scaling=true \
               --data_format=${DATA_FORMAT} \
               --fuse_elewise_add_act_ops=true \
               --fuse_bn_act_ops=true \
               --fuse_bn_add_act_ops=true \
               --enable_addto=${USE_ADDTO} \
               --validate=true \
               --is_profiler=${is_profiler} \
               --profiler_path=${profiler_path} \
               --reader_thread=10 \
               --reader_buf_size=4000 \
               --use_dali=${USE_DALI} \
               --max_iter=${max_iter} \
               --lr=0.1"

    case ${run_mode} in
    sp) train_cmd="python -u train.py "${train_cmd} ;;
    mp)
        rm -rf ./mylog
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --gpus=$CUDA_VISIBLE_DEVICES train.py "${train_cmd}
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
