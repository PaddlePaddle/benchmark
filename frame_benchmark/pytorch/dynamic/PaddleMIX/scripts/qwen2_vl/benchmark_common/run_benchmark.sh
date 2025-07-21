#!/usr/bin/env bash
# Test training benchmark for a model.
# Usage: CUDA_VISIBLE_DEVICES=xxx bash run_benchmark.sh ${model_name} ${run_mode} ${fp_item} ${bs_item} ${max_iter} ${num_workers}

function _set_params(){
    model_item=${1:-"qwen2_vl_sft_2b"}        # (必选) 模型 item |fastscnn|segformer_b0| ocrnet_hrnetw48
    base_batch_size=${2:-"1"}            # (必选) 每张卡上的batch_size
    fp_item=${3:-"bf16"}                 # (必选) fp32|fp16|bf16
    run_process_type=${4:-"MultiP"}      # (必选) 单进程 SingleP|多进程 MultiP
    run_mode=${5:-"DP"}                  # (必选) MP模型并行|DP数据并行|PP流水线并行|混合并行DP1-MP1-PP1|DP1-MP4-PP1
    device_num=${6:-"N1C8"}              # (必选) 使用的卡数量，N1C1|N1C8|N4C8 （4机32卡）
    profiling=${PROFILING:-"false"}      # (必选) Profiling  开关，默认关闭，通过全局变量传递
    model_repo="LLaMA-Factory"             # (必选) 模型套件的名字
    speed_unit="sample/sec"                # (必选)速度指标单位
    skip_steps=2                        # (必选)解析日志，跳过模型前几个性能不稳定的step
    keyword="train_samples_per_second"                       # (必选)解析日志，筛选出性能数据所在行的关键字
    convergence_key="loss"                   # (可选)解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"
    max_epochs=${7:-"1"}                 # （可选）需保证模型执行时间在5分钟内，需要修改代码提前中断的直接提PR 合入套件  或是max_epoch
    num_workers=${8:-"1"}                # (可选)

    # Added for distributed training
    node_num=${9:-"1"}                      #（可选） 节点数量
    node_rank=${10:-"0"}                    # (可选)  节点rank
    master_addr=${11:-"127.0.0.1"}       # (可选) 主节点ip地址
    master_port=${12:-"1928"}               # (可选) 主节点端口号
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
    python analysis_log.py ${model_item} ${log_file} ${speed_log_file} ${device_num} ${base_batch_size} ${fp_item} ${run_process_type}
}

function _train(){
    batch_size=${base_batch_size}  # 如果模型跑多卡但进程时,请在_train函数中计算出多卡需要的bs
    echo "current ${model_name} CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=${device_num}, batch_size=${batch_size}"
    rm -rf work_dirs

    if [ ${model_item} = "qwen2_vl_sft_2b" ];then # 目前只支持2B和7B
        use_model_args="--model_name_or_path Qwen2-VL-2B-Instruct"
        use_output_args="--output_dir work_dirs/qwen2_vl_sft_2b_bs32_1e8"
    else
        use_model_args="--model_name_or_path Qwen2-VL-7B-Instruct"
        use_output_args="--output_dir work_dirs/qwen2_vl_sft_7b_bs32_1e8"
    fi

    train_cmd="src/train.py \
            --do_train \
            ${use_model_args} \
            --stage sft \
            --finetuning_type full \
            --freeze_vision_tower True \
            --deepspeed ds_z2_config.json \
            --dataset chartqa_train_18k \
            --dataset_dir data/ \
            --template qwen2_vl \
            --cutoff_len 8192 \
            --max_samples 2000 \
            --enable_liger_kernel True \
            --overwrite_cache \
            --preprocessing_num_workers 8 \
            ${use_output_args} \
            --logging_steps 1 \
            --save_steps 2000 \
            --save_total_limit 1 \
            --plot_loss \
            --overwrite_output_dir \
            --per_device_train_batch_size ${base_batch_size} \
            --gradient_accumulation_steps 1 \
            --learning_rate 1e-8 \
            --num_train_epochs 1 \
            --lr_scheduler_type cosine \
            --warmup_ratio 0.1 \
            --warmup_steps 100 \
            --weight_decay 0.1 \
            --bf16 \
            --ddp_timeout 9000 \
            --eval_steps -1 \
        "
    case ${run_process_type} in
    SingleP) train_cmd="torchrun --nnodes 1 --nproc_per_node 1 ${train_cmd}" ;;
    MultiP)
    if [ ${device_num:3} = '32' ];then 
        train_cmd="torchrun --nnodes ${node_num} --nproc_per_node 8 --node_rank ${node_rank} --master_addr ${master_addr} --master_port ${master_port} ${train_cmd}"
    else
        train_cmd="torchrun --nnodes 1 --nproc_per_node 8 ${train_cmd}"
    fi;;
    *) echo "choose run_mode(SingleP or MultiP)"; exit 1;
    esac

    timeout 30m ${train_cmd} > ${log_file} 2>&1
    rm -rf work_dirs
    # 这个判断，无论是否成功都是0
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
        export job_fail_flag=1
    else
        echo -e "${model_name}, SUCCESS"
        export job_fail_flag=0
    fi
    }
    # 注释掉，会异常退出
    #kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
    #cd -


_set_params $@
echo "https_proxy $HTTPS_PRO" 
echo "http_proxy $HTTP_PRO" 
export https_proxy=$HTTPS_PRO
export http_proxy=$HTTP_PRO
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com
export frame_version=`python -c "import torch;print(torch.__version__)"`
echo "---------frame_version is torch ${frame_version}"
echo "---------model_branch is ${model_branch}"
echo "---------model_commit is ${model_commit}"
job_bt=`date '+%Y%m%d%H%M%S'`
_train
job_et=`date '+%Y%m%d%H%M%S'`
export model_run_time=$((${job_et}-${job_bt}))
_analysis_log
