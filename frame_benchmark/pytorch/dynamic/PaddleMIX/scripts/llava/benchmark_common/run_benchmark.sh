#!/usr/bin/env bash
# Test training benchmark for a model.
# Usage: CUDA_VISIBLE_DEVICES=xxx bash run_benchmark.sh ${model_name} ${run_mode} ${fp_item} ${bs_item} ${max_iter} ${num_workers}

function _set_params(){
    model_item=${1:-"llava-v1.6-vicuna-7b-sft"}   # (必选) 模型 item |llava-v1.6-vicuna-7b-sft|llava-v1.6-vicuna-13b-sft|llava-v1.6-vicuna-7b-pretrain|llava-v1.6-vicuna-7b-lora_sft|llava-v1.6-vicuna-13b-pretrain|llava-v1.6-vicuna-13b-lora_sft
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

    #模型权重
    if [ ${model_item} = "llava-v1.6-vicuna-7b-sft" ]; then
        model_path="liuhaotian/llava-v1.6-vicuna-7b"
        train_stage="sft"
    fi
    if [ ${model_item} = "llava-v1.6-vicuna-7b-pretrain" ]; then
        model_path="lmsys/vicuna-7b-v1.5"
        train_stage="pretrain"
    fi
    if [ ${model_item} = "llava-v1.6-vicuna-7b-lora_sft" ]; then
        model_path="liuhaotian/llava-v1.6-vicuna-7b"
        train_stage="lora_sft"
    fi
    if [ ${model_item} = "llava-v1.6-vicuna-13b-sft" ]; then
        model_path="liuhaotian/llava-v1.6-vicuna-13b"
        train_stage="sft"
    fi
    if [ ${model_item} = "llava-v1.6-vicuna-13b-pretrain" ]; then
        model_path="lmsys/vicuna-13b-v1.5"
        train_stage="pretrain"
    fi
    if [ ${model_item} = "llava-v1.6-vicuna-13b-lora_sft" ]; then
        model_path="liuhaotian/llava-v1.6-vicuna-13b"
        train_stage="lora_sft"
    fi

    #训练阶段
    if [ ${train_stage} = "sft" ]; then
        train_cmd="llava/train/train_mem.py \
        --deepspeed zero2.json \
        --model_name_or_path ${model_path} \
        --version v1 \
        --data_path /root/.paddlemix/datasets/llava_bench_data/ScienceQA_val_500_torch.json \
        --image_folder /root/.paddlemix/datasets \
        --vision_tower openai/clip-vit-large-patch14-336 \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir ./checkpoints/${model_item} \
        --num_train_epochs 1 \
        --per_device_train_batch_size ${base_batch_size} \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --save_strategy "no" \
        --save_steps 50000 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 0 \
        --lazy_preprocess True
        "
    fi
    #训练阶段
    if [ ${train_stage} = "lora_sft" ]; then
        train_cmd="llava/train/train_mem.py \
        --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
        --deepspeed zero2.json \
        --model_name_or_path ${model_path} \
        --version v1 \
        --data_path /root/.paddlemix/datasets/llava_bench_data/ScienceQA_val_500_torch.json \
        --image_folder /root/.paddlemix/datasets \
        --vision_tower openai/clip-vit-large-patch14-336 \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir ./checkpoints/${model_item} \
        --num_train_epochs 1 \
        --per_device_train_batch_size ${base_batch_size} \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --save_strategy "no" \
        --save_steps 50000 \
        --save_total_limit 1 \
        --learning_rate 2e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 0 \
        --lazy_preprocess True
        "
    fi
    #训练阶段
    if [ ${train_stage} = "pretrain" ]; then
        train_cmd="llava/train/train_mem.py \
        --deepspeed zero2.json \
        --model_name_or_path ${model_path} \
        --version plain \
        --data_path /root/.paddlemix/datasets/llava_bench_data/ScienceQA_val_500_torch.json \
        --image_folder /root/.paddlemix/datasets \
        --vision_tower openai/clip-vit-large-patch14-336 \
        --mm_projector_type mlp2x_gelu \
        --tune_mm_mlp_adapter True \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --bf16 True \
        --output_dir ./checkpoints/${model_item} \
        --num_train_epochs 1 \
        --per_device_train_batch_size ${base_batch_size} \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --save_strategy "no" \
        --save_steps 50000 \
        --save_total_limit 1 \
        --learning_rate 1e-3 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 0 \
        --lazy_preprocess True
        "
    fi
    
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