#!/usr/bin/env bash
# Test training benchmark for a model.
# Usage: CUDA_VISIBLE_DEVICES=xxx bash run_benchmark.sh ${model_name} ${run_mode} ${fp_item} ${bs_item} ${max_iter} ${num_workers}

function _set_params(){
    model_item=${1:-"qwen2_5-7b_sft"}        # (必选) 模型 item |fastscnn|segformer_b0| ocrnet_hrnetw48
    model_name_or_path=${2:-"Qwen/Qwen2.5-1.5B"}    # (必选) 模型名称或路径
    base_batch_size=${3:-"1"}            # (必选) 每张卡上的batch_size
    fp_item=${4:-"bf16"}                 # (必选) fp32|fp16|bf16
    run_stage=${5:-"sft"}                # (必选) sft|lora|dpo
    run_mode=${6:-"DP"}                  # (必选) MP模型并行|DP数据并行|PP流水线并行|混合并行DP1-MP1-PP1|DP1-MP4-PP1
    device_num=${7:-"N1C1"}              # (必选) 使用的卡数量，N1C1|N1C8|N4C8 （4机32卡）
    profiling=${PROFILING:-"false"}      # (必选) Profiling  开关，默认关闭，通过全局变量传递
    model_repo="LLaMA-Factory"             # (必选) 模型套件的名字
    speed_unit="effective_tokens/sec"                # (必选)速度指标单位
    skip_steps=0                        # (必选)解析日志，跳过模型前几个性能不稳定的step
    keyword="effective_tokens_per_sec"                       # (必选)解析日志，筛选出性能数据所在行的关键字
    convergence_key="total_tokens:"                   # (可选)解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"
    max_iter=${8:-"100"}                 # （可选）需保证模型执行时间在5分钟内，需要修改代码提前中断的直接提PR 合入套件  或是max_epoch
    num_workers=${9:-"3"}                # (可选)
    is_large_model=True
    position=${10:-"2"}                  # (可选) 解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"

    # Added for distributed training
    #   以下为通用拼接log路径，无特殊可不用修改
    model_name=${model_item}_bs${base_batch_size}_${fp_item}_${run_mode}  # (必填) 切格式不要改动,与平台页面展示对齐
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}  # （必填） TRAIN_LOG_DIR  benchmark框架设置该参数为全局变量
    profiling_log_path=${PROFILING_LOG_DIR:-$(pwd)}  # （必填） PROFILING_LOG_DIR benchmark框架设置该参数为全局变量
    speed_log_path=${LOG_PATH_INDEX_DIR:-$(pwd)}

    train_log_file=${run_log_path}/${model_repo}_${model_name}_${device_num}_log
    profiling_log_file=${profiling_log_path}/${model_repo}_${model_name}_${device_num}_profiling
    speed_log_file=${speed_log_path}/${model_repo}_${model_name}_${device_num}_speed
    if [ ${profiling} = "true" ];then
            add_options="profiler_options=/"batch_range=[50, 60]; profile_path=model.profile/""
            log_file=${profiling_log_file}
        else
            add_options=""
            log_file=${train_log_file}
    fi
}

function _analysis_log(){
    python analysis_log.py \
            --filename ${log_file} \
            --log_with_profiler ${profiling_log_file:-"not found!"} \
            --profiler_path ${profiler_path:-"not found!"} \
            --speed_log_file ${speed_log_file} \
            --model_name ${model_name} \
            --base_batch_size ${base_batch_size} \
            --run_mode ${run_mode} \
            --fp_item ${fp_item} \
            --keyword ${keyword} \
            --skip_steps ${skip_steps} \
            --device_num ${device_num} \
            --is_large_model ${is_large_model:-"False"} \
            --convergence_key ${convergence_key} \
            --speed_unit ${speed_unit} \
            --position ${position:-2}
}

function _train(){
    batch_size=${base_batch_size}  # 如果模型跑多卡但进程时,请在_train函数中计算出多卡需要的bs
    echo "current ${model_name} CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=${device_num}, batch_size=${batch_size}"
    rm -rf outputs

    case ${run_stage} in
    sft)  train_cmd="llamafactory-cli train benchmark_yaml/${model_item%_*}/sft.yaml" ;;
    lora) train_cmd="llamafactory-cli train benchmark_yaml/${model_item%_*}/lora.yaml" ;;
    dpo)  train_cmd="llamafactory-cli train benchmark_yaml/${model_item%_*}/dpo.yaml" ;;
    *) echo "choose run_stage(sft | lora | dpo)"; exit 1;
    esac


    # 以下为通用执行命令，无特殊可不用修改
    echo "Run with: device_num=${device_num}, run_mode=${run_mode}, run_stage=${run_stage}"
    echo "train_cmd: ${train_cmd}  log_file: ${log_file}"
    export PATH=/opt/torch_native_venv/bin:${PATH}
    export https_proxy=${HTTPS_PRO} 
    export http_proxy=${HTTPS_PRO}
    if [ ${device_num} = "N4C32" ];then
        export MASTER_ADDR=$POD_0_IP
        export MASTER_PORT=36878
        export NODE_RANK=$PADDLE_TRAINER_ID
        export NNODES=$PADDLE_TRAINERS_NUM
        export NPROC_PER_NODE=$PADDLE_TRAINER_COUNT
        unset OMPI_COMM_WORLD_LOCAL_RANK
        export TOKENS_PER_STEP=131072
    fi
    timeout 40m ${train_cmd} > ${log_file} 2>&1
    Tokens_per_second_per_gpu=`cat ${log_file} | grep 'train_samples_per_second =' \
                                            |awk -F'= ' '{print $2}' |awk -F' ' '{print $1}'`
    length=4096
    Total_Tokens_per_second_per_gpu=$(awk -v a="$Tokens_per_second_per_gpu" -v b="$length" 'BEGIN {printf "%.2f\n", a * b}')
    echo "total_tokens: ${Total_Tokens_per_second_per_gpu}" >> ${log_file}
    # 这个判断，无论是否成功都是0
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
        export job_fail_flag=1
    else
        echo -e "${model_name}, SUCCESS"
        export job_fail_flag=0
    fi

    # 注释掉，会异常退出
    #kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
    #cd -
}


_set_params $@
export frame_version=`python -c "import torch;print(torch.__version__)"`
echo "---------frame_version is torch ${frame_version}"
echo "---------model_branch is ${model_branch}"
echo "---------model_commit is ${model_commit}"
job_bt=`date '+%Y%m%d%H%M%S'`
_train
job_et=`date '+%Y%m%d%H%M%S'`
export model_run_time=$((${job_et}-${job_bt}))
_analysis_log