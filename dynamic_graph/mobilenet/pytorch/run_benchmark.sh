#!bin/bash

set -xe
if [[ $# -lt 3 ]]; then
    echo "running job dict is {1: speed, 2:mem, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 1|3|6 sp|mp 1(num_epoch) model_name(MobileNetV1|MobileNetV2)"
    exit
fi

function _set_params(){
    index=$1
    base_batch_size=128                                                       # 单卡的batch_size，如果固定的，可以写死                                                            (必填)
    model_name=${4}                                                          # 当model_name唯一时可写死                                                                          (必填)
    if [ ${model_name} != "MobileNetV1" ] && [ ${model_name} != "MobileNetV2" ] && [ ${model_name} != "resnet" ]; then
        echo "------------> please check the model name!"
        exit 1
    fi
    run_mode=${2}   # Use sp for single GPU and mp for multiple GPUs.
    num_epoch=${3}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi      # 动态图benchmark当前暂未添加profiler，该参数可暂不处理
 
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}

    direction_id=0                                                           # 任务所属方向，0：CV，1：NLP，2：Rec,具体可参考benchmark/scripts/config.ini                        (必填)
    mission_name="图像分类"                                                  # 模型所属任务名称，具体可参考benchmark/scripts/config.ini                                          (必填)
    skip_steps=5                                                             # 解析日志，有些模型前几个step耗时长，需要跳过                                                      (必填)
    keyword="batch_cost:"                                                    # 解析日志，筛选出数据所在行的关键字                                                                (必填)
    separator=" "                                                            # 解析日志，数据所在行的分隔符                                                                      (必填)
    position=9                                                               # 解析日志，按照分隔符分割后形成的数组索引                                                          (必填)
    #range=0:9                                                               # 解析日志，取得列表索引的值后，切片[0：range], 默认最后一位可以不用填, 或者 3:10格式               (选填)
    model_mode=0 # s/step -> samples/s                                       # 解析日志，具体参考benchmark/scripts/analysis.py.                                                  (必填)

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    log_file=${run_log_path}/dynamic_${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/dynamic_${model_name}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_dynamic_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}
}

function _train(){
    if [ ${model_name} = "resnet" ]; then
        model_name_arg="ResNet50"
    else
        model_name_arg=${model_name}
    fi
    train_cmd="--batch_size=${base_batch_size} \
               --workers=10 \
               --data_dir=/work/dygraph/ILSVRC2012_dataset_100 \
               --model=${model_name_arg} \
               --epochs=${num_epoch}"
    if [ ${run_mode} = "sp" ]; then
        train_cmd="python3 -u train_pytorch.py "${train_cmd}
#    else
#        rm -rf ./mylog
#        train_cmd="python3 -m torch.distributed.launch --log_dir=./mylog train.py --use_data_parallel=1 "${train_cmd}
#        log_parse_file="mylog/workerlog.0"
    fi
    ${train_cmd} > ${log_file} 2>&1
#    if [ ${run_mode} != "sp"  -a -d mylog ]; then
#        rm ${log_file}
#        cp mylog/workerlog.0 ${log_file}
#    fi
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_run
