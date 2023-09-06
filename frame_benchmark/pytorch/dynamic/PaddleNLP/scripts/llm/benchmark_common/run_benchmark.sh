#!/usr/bin/env bash

# Test training benchmark for a model.

# Usage: CUDA_VISIBLE_DEVICES=xxx bash run_benchmark.sh ${model_name} ${run_mode} ${fp_item} ${bs_item} ${max_iter} ${num_workers}

function _set_params(){
    model_item=${1:-"huggyllama-llama-7b"}   # (必选) 模型 item |fastscnn|segformer_b0| ocrnet_hrnetw48
    base_batch_size=${2:-"8"}       # (必选) 每张卡上的batch_size
    fp_item=${3:-"fp16"}            # (必选) fp32|fp16
    run_mode=${4:-"DP"}             # (必选) MP模型并行|DP数据并行|PP流水线并行|混合并行DP1-MP1-PP1|DP1-MP4-PP1
    device_num=${5:-"N1C1"}         # (必选) 使用的卡数量，N1C1|N1C8|N4C8 （4机32卡）
    profiling=${PROFILING:-"false"}      # (必选) Profiling  开关，默认关闭，通过全局变量传递
    model_repo="transformers"          # (必选) 模型套件的名字
    speed_unit="tokens/s"          # (必选)速度指标单位
    skip_steps=0                  # (必选)解析日志，跳过模型前几个性能不稳定的step
    keyword="Effective_Tokens_per_second:"                # (必选)解析日志，筛选出性能数据所在行的关键字 
    
    convergence_key="train_loss:"             # (可选)解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"
    model_name_or_path=${6:-"huggyllama-llama-7b"}
    lora=${7:-"false"}
    max_length=${8:-"2048"}
    dataset_name_or_path=${9:-"llm_benchmark_zh"}
    learning_rate=${10:-"3e-05"}
    gradient_checkpointing=${11:-"true"}
    gradient_accumulation_steps=${12:-"8"}
    num_train_epochs=${13:-"1"}
    is_large_model=True           # (可选)普通模型默认为False，如果添加大模型且只取一条ips设置为True

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
    if [ ${profiling} = "true" ];then   # 竞品不要求加profiling
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
        --speed_log_file ${speed_log_file} \
        --model_name ${model_name} \
        --base_batch_size ${base_batch_size} \
        --run_mode ${run_mode} \
        --fp_item ${fp_item} \
        --keyword ${keyword} \
        --skip_steps ${skip_steps} \
        --device_num ${device_num} \
        --is_large_model ${is_large_model} \
        --speed_unit ${speed_unit} \
        --convergence_key ${convergence_key}
}

function _train(){
    batch_size=${base_batch_size}  # 如果模型跑多卡但进程时,请在_train函数中计算出多卡需要的bs

    echo "current ${model_name} CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=${device_num}, batch_size=${batch_size}"

    train_options="    --model_name_or_path ${model_name_or_path} \
            --dataset_name_or_path ${dataset_name_or_path} \
            --per_device_train_batch_size ${base_batch_size} \
            --output_dir output \
            --gradient_accumulation_steps ${gradient_accumulation_steps} \
            --fp16 1 \
            --fp16_opt_level O2 \
            --num_train_epochs ${num_train_epochs} \
            --learning_rate ${learning_rate} \
            --evaluation_strategy no \
            --save_strategy no \
            --src_length 1024 \
            --max_length ${max_length} \
            --do_train 1 \
            --do_eval 0 \
            --gradient_checkpointing ${gradient_checkpointing} \
            --lora ${lora}"

    if [ "N1C1" = ${device_num} ]; then
        train_cmd="python benchmark.py ${train_options}"
    else
        train_cmd="python -m torch.distributed.run --nproc_per_node=8 benchmark.py --deepspeed ds_config_stage3.json ${train_options}" ;
    fi
    
    echo "train_cmd: ${train_cmd}  log_file: ${log_file}"
    ${train_cmd} > ${log_file} 2>&1

    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
    else
        echo -e "${model_name}, SUCCESS"
    fi
    #kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
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