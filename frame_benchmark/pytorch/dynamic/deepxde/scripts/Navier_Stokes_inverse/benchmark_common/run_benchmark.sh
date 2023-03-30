#!/usr/bin/env bash

# Test training benchmark for a model.
function _set_params(){
    model_item=${1:-"model_item"}   # (必选) 模型 item |fastscnn|segformer_b0| ocrnet_hrnetw48
    base_batch_size=${2:-"1"}       # (必选) 如果是静态图单进程，则表示每张卡上的BS，需在训练时*卡数
    fp_item=${3:-"fp32"}            # (必选) fp32|fp16
    run_mode=${4:-"DP"}             # (必选) MP模型并行|DP数据并行|PP流水线并行|混合并行DP1-MP1-PP1|DP1-MP4-PP1
    device_num=${5:-"N1C1"}         # (必选) 使用的卡数量，N1C1|N1C8|N4C32 （4机32卡）
 
    backend="paddle"
    model_repo="deepxde"          # (必选) 模型套件的名字
    speed_unit="samples/sec"         # (必选)速度指标单位
    skip_steps=0                  # (必选)解析日志，跳过模型前几个性能不稳定的step
    keyword="ips:"                 # (必选)解析日志，筛选出性能数据所在行的关键字
    convergence_key="loss:"        # (可选)解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"

#   以下为通用执行命令，无特殊可不用修改
    model_name=${model_item}_bs${base_batch_size}_${fp_item}_${run_mode}  # (必填) 且格式不要改动,与竞品名称对齐
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}  # （必填） TRAIN_LOG_DIR  benchmark框架设置该参数为全局变量
    speed_log_path=${LOG_PATH_INDEX_DIR:-$(pwd)}
    # deepxde_Eular_beam_bs2_fp32_DP_N1C1_log
    train_log_file=${run_log_path}/${model_repo}_${model_name}_${device_num}_log
    speed_log_file=${speed_log_path}/${model_repo}_${model_name}_${device_num}_speed
}

function _analysis_log(){
    echo "train_log_file: ${train_log_file}"
    echo "speed_log_file: ${speed_log_file}"
    cmd="python analysis_log.py --filename ${train_log_file} \
        --speed_log_file ${speed_log_file} \
        --model_name ${model_name} \
        --base_batch_size ${base_batch_size} \
        --run_mode ${run_mode} \
        --fp_item ${fp_item} \
        --keyword ${keyword} \
        --skip_steps ${skip_steps} \
        --device_num ${device_num} "
    echo ${cmd}
    eval $cmd
}

function _train(){
    echo "current CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, model_name=${model_name}, device_num=${device_num}, is profiling=${profiling}"

#   以下为通用执行命令，无特殊可不用修改

    export DDE_BACKEND=pytorch
    train_cmd="python3.7 examples/pinn_forward/Euler_beam.py" 

    echo "train_cmd: ${train_cmd}  log_file: ${train_log_file}"
    timeout 15m ${train_cmd} > ${train_log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
    else
        echo -e "${model_name}, SUCCESS"
    fi
    #kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
}

_set_params $@
export frame_version=`python -c "import torch;print(torch.__version__)"`
echo "---------frame_version is ${frame_version}"
echo "---------Model commit is ${model_commit}"
echo "---------model_branch is ${model_branch}"

job_bt=`date '+%Y%m%d%H%M%S'`
_train
job_et=`date '+%Y%m%d%H%M%S'`
export model_run_time=$((${job_et}-${job_bt}))
_analysis_log     
