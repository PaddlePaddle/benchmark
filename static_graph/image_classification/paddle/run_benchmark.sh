#!bin/bash
set -xe

if [[ $# -lt 4 ]]; then
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh speed|mem|maxbs 32 model_name sp|mp /ssd1/ljh/logs"
    exit
fi

function _set_params(){
    index=$1                         # 速度(speed)|显存占用(mem)|单卡最大支持batch_size(maxbs)(必填)
    base_batch_size=$2               # 单卡的batch_size，如果固定的，可以写死（必填）
    model_name=$3                    # 模型名字如："SE-ResNeXt50"，如果是固定的，可以写死，如果需要其他参数可以参考bert实现（必填）
    run_mode=${4:-"sp"}              # 单进程(sp)|多进程(mp)，默认单进程（必填）
    run_log_root=${5:-$(pwd)}        # 训练保存的日志目录（必填）

    skip_steps=2                     # 解析日志，有些模型前几个step耗时长，需要跳过(必填)
    keyword="elapse"                 # 解析日志，筛选出数据所在行的关键字(必填)
    separator=" "                    # 解析日志，数据所在行的分隔符(必填)
    position=-2                      # 解析日志，按照分隔符分割后形成的数组索引(必填)
    model_mode=0                     # 解析日志，若数据单位是s/step，则为0，若数据单位是step/s,则为1(必填)
    #range=-1                        # 解析日志，取得列表索引的值后，切片[0：range], 默认最后一位可以不用填

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    if [ $run_mode = "sp" ]; then
        batch_size=`expr $base_batch_size \* $num_gpu_devices`
    else
        batch_size=$base_batch_size
    fi
    log_file=${run_log_root}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
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
    if echo {ResNet50 ResNet101} | grep -w $model_name &>/dev/null
    then
       train_cmd="  --model=${model_name} \
           --batch_size=${batch_size} \
           --total_images=1281167 \
           --class_dim=1000 \
           --image_shape=3,224,224 \
           --model_save_dir=output/ \
           --lr_strategy=piecewise_decay \
           --num_epochs=${num_epochs} \
           --lr=0.1 \
           --l2_decay=1e-4"
    elif echo {SE_ResNeXt50_32x4d} | grep -w $model_name &>/dev/null
    then
        train_cmd=" --model=${model_name} \
           --batch_size=${batch_size} \
           --total_images=1281167 \
           --class_dim=1000 \
           --image_shape=3,224,224 \
           --model_save_dir=output/ \
           --data_dir=data/ILSVRC2012 \
           --lr_strategy=cosine_decay \
           --lr=0.1 \
           --l2_decay=1.2e-4 \
           --num_epochs=${num_epochs}"
    else
        echo "model: $model_name not support!"
	exit
    fi

    case ${run_mode} in
    sp) train_cmd="python -u train.py "${train_cmd} ;;
    mp)
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --selected_gpus=$CUDA_VISIBLE_DEVICES train.py "${train_cmd}
        log_parse_file="mylog/workerlog.0" ;;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac

    ${train_cmd} > ${log_file} 2>&1 &
    train_pid=$!
    sleep 300
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
