#!bin/bash
set -xe

if [[ $# -lt 1 ]]; then
    echo "running job dict is {1: speed, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 1|3|6 sp|mp 1000(max_iter)"
    exit
fi

function _set_params(){
    index="$1"
    run_mode=${2:-"sp"}
    max_iter=${3}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi

    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}

    base_batch_size=16
    model_name="StarGAN"_bs${base_batch_size}
    mission_name="图像生成"           # 模型所属任务名称，具体可参考scripts/config.ini                              （必填）
    direction_id=0                    # 任务所属方向，0：CV，1：NLP，2：Rec。                                       （必填）
    skip_steps=5                      # 解析日志，有些模型前几个step耗时长，需要跳过                                （必填）
    keyword="ips:"                    # 解析日志，筛选出数据所在行的关键字                                          （必填）
    ips_unit="images/s"

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}
    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/${model_name}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}
}

function _set_env(){
    #打开后速度变快
    export FLAGS_cudnn_exhaustive_search=1
    #显存占用减少，不影响性能
    export FLAGS_eager_delete_tensor_gb=0.0
    export FLAGS_conv_workspace_size_limit=256
}

function _train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=${base_batch_size}"
    train_cmd=" --model_net ${model_name%_bs*} \
        --dataset celeba \
        --crop_size 178 \
        --image_size 128 \
        --train_list ./data/celeba/list_attr_celeba.txt \
        --gan_mode wgan \
        --batch_size ${base_batch_size} \
        --epoch 20 \
        --profile=${is_profiler} \
        --profiler_path=${profiler_path} \
        --max_iter=${max_iter} \
        --n_critic 1 \
        --run_test False"

    case ${run_mode} in
    sp)
        train_cmd="python -u train.py "${train_cmd}
        ;;
    mp)
        rm -rf ./mylog
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --gpus=$CUDA_VISIBLE_DEVICES train.py "${train_cmd}
        log_parse_file="mylog/workerlog.0"
        ;;
    *)
        echo "choose run_mode: sp or mp"
        exit 1
        ;;
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
