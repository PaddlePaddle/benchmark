#!/usr/bin/env bash
set -xe
# 运行示例：CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} ${model_mode} ${config_path}
# 参数说明
function _set_params(){
    run_mode=${1:-"sp"}          # 单卡sp|多卡mp
    batch_size=${2:-"64"}
    fp_item=${3:-"fp32"}        # fp32|fp16
    model_item=${4:-"model_item"}
    config_path=${5:-"config_path"}
    run_log_path="${TRAIN_LOG_DIR:-$(pwd)}"  # TRAIN_LOG_DIR 后续QA设置该参
 
#   以下不用修改   
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    log_file=${run_log_path}/${model_item}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}.log

    case ${model_item} in       # model_name 在analysis 里面拼接fp以及bs，构成json格式
    MobileNetV2) model_name=MobileNetV2;;
    ShuffleNetV2) model_name=ShuffleNetV2_x1_0;;
    SwinTransformer)  model_name=SwinTransformer_tiny_patch4_window7_224;;
    MobileNetV3Large1.0) model_name=MobileNetV3_large_x1_0;;
    *) echo "set your model by MobileNetV2|ShuffleNetV2|SwinTransformer|MobileNetV3Large1.0"; exit 1;
    esac
}
function _train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"

    case ${run_mode} in
    sp) train_cmd="python tools/train.py ${config_path} --no-validate --cfg-options data.samples_per_gpu=${batch_size} log_config.interval=10";;
    mp)
	train_cmd="python -m torch.distributed.launch --nproc_per_node=${num_gpu_devices} --master_port=29500 ./tools/train.py ${config_path} --no-validate --cfg-options data.samples_per_gpu=${batch_size} log_config.interval=10 --launcher pytorch";;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac

# 以下不用修改
    timeout 15m ${train_cmd}  > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_item}, FAIL"
        export job_fail_flag=1
    else
        echo -e "${model_item}, SUCCESS"
        export job_fail_flag=0
    fi
    kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
 
}

_set_params $@
rm -rf work_dirs
#_train
source ${ROOT_DIR}/scripts/run_model.sh
_run
python analysis_log.py -d work_dirs -m ${model_item} -b ${batch_size} -n ${num_gpu_devices}
eval "mv work_dirs ${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}"