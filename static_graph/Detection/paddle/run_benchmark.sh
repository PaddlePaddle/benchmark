#!bin/bash
set -xe
if [[ $# -lt 3 ]]; then
    echo "running job dict is {1: speed, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1|3|6 model_name sp|mp 1000(max_iter)"
    exit
fi

function _set_params(){
    index=$1                         # 速度(speed)|显存占用(mem)|单卡最大支持batch_size(maxbs)                       (必填)
    base_batch_size=2                # 单卡的batch_size，如果固定的，可以写死                                        （必填）
    model_name=$2                    # 模型名字如："SE-ResNeXt50"，如果是固定的，可以写死，如果需要其他参数可以参考bert实现（必填）
    mission_name="目标检测"          # 模型所属任务名称，具体可参考scripts/config.ini                                （必填）
    direction_id=0                   # 任务所属方向，0：CV，1：NLP，2：Rec。                                         (必填)
    run_mode=${3:-"sp"}              # 单进程(sp)|多进程(mp)，默认单进程                                            （必填）
    max_iter=${4}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi

    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}
    skip_steps=2                     # 解析日志，有些模型前几个step耗时长，需要跳过                                    (必填)
    keyword="ips:"                   # 解析日志，筛选出数据所在行的关键字                                             (必填)
    model_mode=-1                    # 解析日志，具体参考scripts/analysis.py.                                      (必填)
    ips_unit="images/s"
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    if [ ${model_name} = "mask_rcnn_fpn_resnet" ] || [ ${model_name} = "cascade_rcnn_fpn" ]; then
        base_learning_rate=0.002 # 与竞品保持一致
    elif [ ${model_name} = "mask_rcnn_fpn_resnext" ] || [ ${model_name} = "retinanet_rcnn_fpn" ]; then
        base_learning_rate=0.001 # 与竞品保持一致
    else
        echo "model_name must be mask_rcnn_fpn_resnet | mask_rcnn_fpn_resnext | retinanet_rcnn_fpn | cascade_rcnn_fpn"
        exit 1
    fi
    if [[ ${run_mode} = "sp" ]]; then
        batch_size=`expr $base_batch_size \* $num_gpu_devices`
        learning_rate=$(awk 'BEGIN{ print '${base_learning_rate}' * '${num_gpu_devices}' }')
    else
        batch_size=$base_batch_size
        learning_rate=${base_learning_rate}
    fi
    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/${model_name}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}

    if [[ ${model_name} = "mask_rcnn_fpn_resnet" ]]; then
        config_file="configs/mask_rcnn_r101_vd_fpn_1x.yml"
    elif [[ ${model_name} = "mask_rcnn_fpn_resnext" ]];then
        config_file="configs/mask_rcnn_x101_vd_64x4d_fpn_1x.yml"
    elif [[ ${model_name} = "retinanet_rcnn_fpn" ]];then
        config_file="configs/retinanet_r50_fpn_1x.yml"
    elif [[ ${model_name} = "cascade_rcnn_fpn" ]];then
        config_file="configs/cascade_rcnn_r50_fpn_1x.yml"
    else
        echo "model_name must be mask_rcnn_fpn_resnet | mask_rcnn_fpn_resnext | retinanet_rcnn_fpn | cascade_rcnn_fpn"
        exit 1
    fi
    model_name=${model_name}_bs${base_batch_size}
}

function _set_env(){
    #开启gc
    export FLAGS_eager_delete_tensor_gb=0.0
    export FLAGS_fraction_of_gpu_memory_to_use=0.98
    export FLAGS_conv_workspace_size_limit=4096
}

function _train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"

    train_cmd="-c ${config_file}  \
               --opt LearningRate.base_lr=${learning_rate} TrainReader.batch_size=${base_batch_size} max_iters=${max_iter} \
               --is_profiler=${is_profiler} \
               --profiler_path=${profiler_path}"

    case ${run_mode} in
    sp) train_cmd="python -u tools/train.py "${train_cmd} ;;
    mp)
        rm -rf ./mylog
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --gpus=$CUDA_VISIBLE_DEVICES tools/train.py "${train_cmd}
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
}


source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_set_env
_run

