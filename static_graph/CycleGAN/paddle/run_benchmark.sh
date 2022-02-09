#!bin/bash
set -xe

if [[ $# -lt 1 ]]; then
    echo "running job dict is {1: speed, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 1|3 sp|mp 1000(max_iter)"
    exit
fi

function _set_params(){
    index="$1"
    run_mode=${2:-"sp"}
    max_iter=${3}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi

    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}

    base_batch_size=1
    model_name="CycleGAN"_bs${base_batch_size}
    mission_name="图像生成"           # 模型所属任务名称，具体可参考scripts/config.ini                        （必填）
    direction_id=0                    # 任务所属方向，0：CV，1：NLP，2：Rec。                                  (必填)
    skip_steps=5                      # 解析日志，有些模型前几个step耗时长，需要跳过                           (必填)
    keyword="ips:"                    # 解析日志，筛选出数据所在行的关键字                                     (必填)
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

    timeout 15m python train.py --model_net ${model_name%_bs*} \
                    --dataset cityscapes \
                    --batch_size ${base_batch_size} \
                    --net_G resnet_9block \
                    --g_base_dim 32 \
                    --net_D basic \
                    --epoch 3 \
                    --run_test False \
                    --image_size 286 \
                    --crop_size 256 \
                    --crop_type Random \
                    --profile=${is_profiler} \
                    --profiler_path=${profiler_path} \
                    --max_iter=${max_iter} \
                    --output ./output/cyclegan/ > ${log_file} 2>&1

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