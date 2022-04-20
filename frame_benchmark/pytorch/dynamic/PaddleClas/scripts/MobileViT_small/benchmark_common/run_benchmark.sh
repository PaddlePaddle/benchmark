#!/usr/bin/env bash

# Test training benchmark for a model.
declare -A dic
dic=( ["MobileNetV2_fp32"]="configs/mobilenet_v2/mobilenet_v2_b32x8_imagenet.py"
      ["ShuffleNetV2_x1_0_fp32"]="configs/shufflenet_v2/shufflenet_v2_1x_b64x16_linearlr_bn_nowd_imagenet.py"
      ["SwinTransformer_tiny_patch4_window7_224_fp32"]="configs/swin_transformer/swin_base_224_b16x64_300e_imagenet.py"
      ["MobileNetV3_large_x1_0_fp32"]="configs/mobilenet_v3/mobilenet_v3_large_imagenet.py"
      ["ResNet50_fp32"]="configs/resnet/resnet50_b32x8_imagenet.py"
      ["ResNet152_fp32"]="configs/resnet/resnet152_b32x8_imagenet.py"
      ["ResNet50_fp16"]="configs/fp16/resnet50_b32x8_fp16_imagenet.py"
	)

# Usage: CUDA_VISIBLE_DEVICES=xxx bash run_benchmark.sh ${model_name} ${run_mode} ${fp_item} ${bs_item} ${max_epochs} ${num_workers}

function _set_params(){
    model_item=${1:-"model_item"}   # (必选) 模型 item |fastscnn|segformer_b0| ocrnet_hrnetw48
    base_batch_size=${2:-"2"}       # (必选) 每张卡上的batch_size
    fp_item=${3:-"fp32"}            # (必选) fp32|fp16
    run_mode=${4:-"DP"}             # (必选) MP模型并行|DP数据并行|PP流水线并行|混合并行DP1-MP1-PP1|DP1-MP4-PP1
    device_num=${5:-"N1C1"}         # (必选) 使用的卡数量，N1C1|N1C8|N4C8 （4机32卡）
    profiling=${PROFILING:-"false"}      # (必选) Profiling  开关，默认关闭，通过全局变量传递
    model_repo="ml-cvnets"          # (必选) 模型套件的名字
    ips_unit="samples/sec"         # (必选)速度指标单位
    skip_steps=10                  # (必选)解析日志，跳过模型前几个性能不稳定的step
    keyword="ips:"                 # (必选)解析日志，筛选出性能数据所在行的关键字

    convergence_key=""             # (可选)解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"
    max_epochs=${7:-"1"}                # （可选）需保证模型执行时间在5分钟内，需要修改代码提前中断的直接提PR 合入套件  或是max_epoch
    num_workers=${8:-"4"}             # (可选)

    #   以下为通用拼接log路径，无特殊可不用修改
    model_name=${model_item}_bs${base_batch_size}_${fp_item}_${run_mode}  # (必填) 切格式不要改动,与平台页面展示对齐
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}  # （必填） TRAIN_LOG_DIR  benchmark框架设置该参数为全局变量
    profiling_log_path=${PROFILING_LOG_DIR:-$(pwd)}  # （必填） PROFILING_LOG_DIR benchmark框架设置该参数为全局变量
    speed_log_path=${LOG_PATH_INDEX_DIR:-$(pwd)}
    # mmsegmentation_fastscnn_bs2_fp32_MultiP_DP_N1C1_log
    train_log_file=${run_log_path}/${model_repo}_${model_name}_${device_num}_log
    profiling_log_file=${profiling_log_path}/${model_repo}_${model_name}_${device_num}_profiling
    speed_log_file=${speed_log_path}/${model_repo}_${model_name}_${device_num}_speed
    if [ ${profiling} = "true" ];then
            add_options="profiler_options=\"batch_range=[50, 60]; profile_path=model.profile\""
            log_file=${profiling_log_file}
        else
            add_options=""
            log_file=${train_log_file}
    fi
}

function _analysis_log(){
    python analysis_log.py -d ${log_file} -m ${model_item} -b ${batch_size} -n ${device_num} -s ${speed_log_file} -f ${fp_item}
}

function _train(){
    batch_size=${base_batch_size}  # 如果模型跑多卡但进程时,请在_train函数中计算出多卡需要的bs

    echo "current ${model_name} CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=${device_num}, batch_size=${batch_size}"
    train_config="config/classification/mobilevit_small.yaml"

#    sed -ri 's/(mixed_precision: ).*/\1flase/' config/classification/mobilevit_small.yaml
    sed -ri 's/(root_train: ")[^"]*/\1\.\/ILSVRC2012_val/' config/classification/mobilevit_small.yaml
    sed -ri 's/(root_val: ")[^"]*/\1\.\/ILSVRC2012_val/' config/classification/mobilevit_small.yaml
    sed -ri "s/(max_epochs: )[0-9]*/\1${max_epochs}/" config/classification/mobilevit_small.yaml
    sed -ri "s/(train_batch_size0: )[0-9]*/\1${batch_size}/" config/classification/mobilevit_small.yaml
    sed -ri "s/(^\s*workers: )[0-9]*/\1${num_workers}/" config/classification/mobilevit_small.yaml
    sed -ri '/ema/{n;s/(enable: )true/\1flase/;}' config/classification/mobilevit_small.yaml

    sed -ri '/sampler/{n;s/(name: ")[^"]*/\1batch_sampler/;}' config/classification/mobilevit_small.yaml
    sed -ri 's/vbs:/bs:/'  config/classification/mobilevit_small.yaml
    sed -ri 's/^.*max_n_scales.*$//'  config/classification/mobilevit_small.yaml
    sed -ri 's/^.*min_crop_size_width.*//'  config/classification/mobilevit_small.yaml
    sed -ri 's/^.*max_crop_size_width.*//'  config/classification/mobilevit_small.yaml
    sed -ri 's/^.*min_crop_size_height.*//'  config/classification/mobilevit_small.yaml
    sed -ri 's/^.*max_crop_size_height.*//'  config/classification/mobilevit_small.yaml
    sed -ri 's/^.*check_scale.*//'  config/classification/mobilevit_small.yaml

    case ${device_num} in
    N1C1) sed -ri '/ddp/{n;s/(enable: )true/\1flase/;}' config/classification/mobilevit_small.yaml ;;
    N1C8) sed -ri 's/(^\s*dist_port: )[0-9]*/\129500/' config/classification/mobilevit_small.yaml ;;
    *) echo "choose run_process_type(SingleP or MultiP)"; exit 1;
    esac

    train_cmd="python -u main_train.py --common.config-file ${train_config} --common.results-loc results_mobilevit_small"

#   以下为通用执行命令，无特殊可不用修改
    timeout 15m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
    else
        echo -e "${model_name}, SUCCESS"
    fi
}

_set_params $@
# export model_branch=`git symbolic-ref HEAD 2>/dev/null | cut -d"/" -f 3`
# export model_commit=$(git log|head -n1|awk '{print $2}')
export frame_version=`python -c "import torch;print(torch.__version__)"`
echo "---------frame_version is torch ${frame_version}"
echo "---------model_branch is ${model_branch}"
echo "---------model_commit is ${model_commit}"

job_bt=`date '+%Y%m%d%H%M%S'`
_train
job_et=`date '+%Y%m%d%H%M%S'`
export model_run_time=$((${job_et}-${job_bt}))
_analysis_log
