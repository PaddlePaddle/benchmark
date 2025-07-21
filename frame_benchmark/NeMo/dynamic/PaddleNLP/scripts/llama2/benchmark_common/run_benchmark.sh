#!/usr/bin/env bash
# Test training benchmark for a model.
# Usage: CUDA_VISIBLE_DEVICES=xxx bash run_benchmark.sh ${model_name} ${run_mode} ${fp_item} ${bs_item} ${max_iter} ${num_workers}

function _set_params(){
    model_name_or_path=${model_name_or_path:-"Qwen/Qwen2.5-1.5B"}    # (必选) 模型名称或路径
    tensor_model_parallel_size=${tensor_model_parallel_size:-"1"}    # (必选) tp_size
    pipeline_model_parallel_size=${pipeline_model_parallel_size:-"1"}    # (必选) pp_size
    sequence_parallel=${sequence_parallel:-"True"}    
    recompute=${recompute:-none}    
    activations_checkpoint_method=${activations_checkpoint_method:-"null"}    
    recompute_layers=${recompute_layers:-"5"}    
    scheme=${scheme:-none}   
    model_item=${model_item:-"qwen2_5-7b_sft"}        # (必选) 模型 item |fastscnn|segformer_b0| ocrnet_hrnetw48
    base_batch_size=${bs_item:-"1"}            # (必选) 每张卡上的batch_size
    fp_item=${fp_item:-"bf16"}                 # (必选) fp32|fp16|bf16
    run_stage=${run_stage:-"sft"}                # (必选) sft|lora|dpo
    run_mode=${run_mode:-"DP"}                  # (必选) MP模型并行|DP数据并行|PP流水线并行|混合并行DP1-MP1-PP1|DP1-MP4-PP1
    device_num=${device_num:-"N1C1"}              # (必选) 使用的卡数量，N1C1|N1C8|N4C8 （4机32卡）
    profiling=${PROFILING:-"false"}      # (必选) Profiling  开关，默认关闭，通过全局变量传递
    model_repo="NeMo"             # (必选) 模型套件的名字
    speed_unit="effective_tokens/sec"                # (必选)速度指标单位
    skip_steps=0                        # (必选)解析日志，跳过模型前几个性能不稳定的step
    keyword="effective_tokens_per_second_per_device:"                       # (必选)解析日志，筛选出性能数据所在行的关键字
    convergence_key="total_tokens:"                   # (可选)解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"
    is_large_model=True
    position=${position:-"2"}                  # (可选) 解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"

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
            --speed_unit ${speed_unit} 
            # --position ${position:-2}
}

function _train(){
    batch_size=${base_batch_size}  # 如果模型跑多卡但进程时,请在_train函数中计算出多卡需要的bs
    echo "current ${model_name} CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=${device_num}, batch_size=${batch_size}"
    
    MODEL="/opt/models/${model_name_or_path}.nemo"
    experiment_name="${model_name_or_path}-${run_stage}"
    OUTPUT_DIR="/opt/nemo-benchmark/logs/${model_name_or_path}"
    TRAIN_DS=[/opt/nemo-benchmark/data-Llama2/sft/train/packed_4096_seed0.npy]
    VALID_DS=[/opt/nemo-benchmark/data-Llama2/sft/train/packed_4096_seed0.npy]

    TRAIN_DATA_PATH="/opt/nemo-benchmark/data-Llama2/nemo_dpo_train.jsonl"
    VALID_DATA_PATH="/opt/nemo-benchmark/data-Llama2/nemo_dpo_dev.jsonl"

    PORT=36789 # 端口号
    SCHEME="none"

    train_cmd="exp_manager.exp_dir=${OUTPUT_DIR} \
    exp_manager.explicit_log_dir=${OUTPUT_DIR} \
    ++exp_manager.log_step_performance=True \
    ++exp_manager.create_mlflow_logger=True \
    ++exp_manager.mlflow_logger_kwargs.experiment_name=${experiment_name} \
    ++exp_manager.mlflow_logger_kwargs.prefix=${experiment_name} \
    trainer.devices=$PADDLE_TRAINER_COUNT \
    trainer.num_nodes=$PADDLE_TRAINERS_NUM \
    model.tensor_model_parallel_size=${tensor_model_parallel_size} \
    model.sequence_parallel=${sequence_parallel} \
    model.pipeline_model_parallel_size=${pipeline_model_parallel_size} \
    model.activations_checkpoint_granularity=${recompute} \
    model.activations_checkpoint_method=${activations_checkpoint_method} \
    model.activations_checkpoint_num_layers=${recompute_layers} \
    model.micro_batch_size=1 \
    model.global_batch_size=32 \
    ++model.data.train_ds.packed_sequence=True \
    model.data.train_ds.max_seq_length=4096 \
    model.data.train_ds.file_names=${TRAIN_DS} \
    model.data.train_ds.concat_sampling_probabilities=[1.0] \
    model.data.validation_ds.file_names=${VALID_DS} \
    ++model.data.validation_ds.packed_sequence=True \
    model.data.validation_ds.max_seq_length=4096 \
    ++model.peft.peft_scheme=${scheme} \
            "

    case ${run_stage} in
    sft)  train_cmd=" \
    python -m torch.distributed.launch \
        --use-env \
        --nproc-per-node=$PADDLE_TRAINER_COUNT \
        --nnodes=$PADDLE_TRAINERS_NUM \
        --node-rank=$PADDLE_TRAINER_ID \
        --master-addr=$POD_0_IP \
        --master-port=$PORT \
        /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py \
        --config-path=/opt/nemo-benchmark/conf_sft \
        --config-name=${model_name_or_path}.yaml \
        exp_manager.exp_dir=${OUTPUT_DIR} \
        exp_manager.explicit_log_dir=${OUTPUT_DIR} \
        ++exp_manager.log_step_performance=True \
        ++exp_manager.create_mlflow_logger=True \
        ++exp_manager.mlflow_logger_kwargs.experiment_name=${experiment_name} \
        ++exp_manager.mlflow_logger_kwargs.prefix=${experiment_name} \
        trainer.devices=$PADDLE_TRAINER_COUNT \
        trainer.num_nodes=$PADDLE_TRAINERS_NUM \
        model.tensor_model_parallel_size=${tensor_model_parallel_size} \
        model.sequence_parallel=${sequence_parallel} \
        model.pipeline_model_parallel_size=${pipeline_model_parallel_size} \
        model.activations_checkpoint_granularity=${recompute} \
        model.activations_checkpoint_method=${activations_checkpoint_method} \
        model.activations_checkpoint_num_layers=${recompute_layers} \
        model.micro_batch_size=1 \
        model.global_batch_size=32 \
        ++model.data.train_ds.packed_sequence=True \
        model.data.train_ds.max_seq_length=4096 \
        model.data.train_ds.file_names=${TRAIN_DS} \
        model.data.train_ds.concat_sampling_probabilities=[1.0] \
        model.data.validation_ds.file_names=${VALID_DS} \
        ++model.data.validation_ds.packed_sequence=True \
        model.data.validation_ds.max_seq_length=4096 \
        ++model.peft.peft_scheme=${scheme}
    " ;;
    lora)  train_cmd=" \
    python -m torch.distributed.launch \
        --use-env \
        --nproc-per-node=$PADDLE_TRAINER_COUNT \
        --nnodes=$PADDLE_TRAINERS_NUM \
        --node-rank=$PADDLE_TRAINER_ID \
        --master-addr=$POD_0_IP \
        --master-port=$PORT \
        /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py \
        --config-path=/opt/nemo-benchmark/conf_sft \
        --config-name=${model_name_or_path}.yaml \
        exp_manager.exp_dir=${OUTPUT_DIR} \
        exp_manager.explicit_log_dir=${OUTPUT_DIR} \
        ++exp_manager.log_step_performance=True \
        ++exp_manager.create_mlflow_logger=True \
        ++exp_manager.mlflow_logger_kwargs.experiment_name=${experiment_name} \
        ++exp_manager.mlflow_logger_kwargs.prefix=${experiment_name} \
        trainer.devices=$PADDLE_TRAINER_COUNT \
        trainer.num_nodes=$PADDLE_TRAINERS_NUM \
        model.tensor_model_parallel_size=${tensor_model_parallel_size} \
        model.sequence_parallel=False \
        model.pipeline_model_parallel_size=${pipeline_model_parallel_size} \
        model.virtual_pipeline_model_parallel_size=1 \
        model.overlap_p2p_comm=False \
        model.activations_checkpoint_granularity=${recompute} \
        model.activations_checkpoint_method=${activations_checkpoint_method} \
        model.activations_checkpoint_num_layers=${recompute_layers} \
        model.micro_batch_size=1 \
        model.global_batch_size=32 \
        ++model.data.train_ds.packed_sequence=True \
        model.data.train_ds.max_seq_length=4096 \
        model.data.train_ds.file_names=${TRAIN_DS} \
        model.data.train_ds.concat_sampling_probabilities=[1.0] \
        model.data.validation_ds.file_names=${VALID_DS} \
        ++model.data.validation_ds.packed_sequence=True \
        model.data.validation_ds.max_seq_length=4096 \
        ++model.peft.peft_scheme=${scheme} \
        ++model.peft.lora_tuning.target_modules=['attention','mlp'] \
    " ;;
    dpo)  train_cmd=" \
    python -m torch.distributed.launch \
        --use-env \
        --nproc-per-node=$PADDLE_TRAINER_COUNT \
        --nnodes=$PADDLE_TRAINERS_NUM \
        --node-rank=$PADDLE_TRAINER_ID \
        --master-addr=$POD_0_IP \
        --master-port=$PORT \
        /opt/NeMo-Aligner/examples/nlp/gpt/train_gpt_dpo.py \
        --config-path=/opt/nemo-benchmark/conf_dpo \
        --config-name=${model_name_or_path}.yaml \
        exp_manager.exp_dir=${OUTPUT_DIR} \
        exp_manager.explicit_log_dir=${OUTPUT_DIR} \
        ++exp_manager.log_step_performance=True \
        ++exp_manager.create_mlflow_logger=True \
        ++exp_manager.mlflow_logger_kwargs.experiment_name=${experiment_name} \
        ++exp_manager.mlflow_logger_kwargs.prefix=${experiment_name} \
        trainer.devices=$PADDLE_TRAINER_COUNT \
        trainer.num_nodes=$PADDLE_TRAINERS_NUM \
        ++model.tensor_model_parallel_size=${tensor_model_parallel_size} \
        model.sequence_parallel=False \
        ++model.pipeline_model_parallel_size=${pipeline_model_parallel_size} \
        model.virtual_pipeline_model_parallel_size=1 \
        model.overlap_p2p_comm=False \
        ++model.activations_checkpoint_granularity=${recompute} \
        ++model.activations_checkpoint_method=${activations_checkpoint_method} \
        ++model.activations_checkpoint_num_layers=${recompute_layers} \
        ++model.micro_batch_size=1 \
        ++model.global_batch_size=32 \
        ++model.encoder_seq_length=4096 \
        'model.data.data_prefix={train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${VALID_DATA_PATH}]}' \
        exp_manager.create_wandb_logger=false \
        exp_manager.wandb_logger_kwargs.project=dpo_training \
        exp_manager.wandb_logger_kwargs.name=dpo_training \
        exp_manager.explicit_log_dir=${OUTPUT_DIR} \
        ++trainer.dpo.max_epochs=1 \
        ++trainer.dpo.max_steps=20 \
        ++trainer.dpo.save_interval=1000 \
        ++trainer.dpo.val_check_interval=1000 \
        ++trainer.dpo.limit_val_batches=0.1 \
        ++model.dpo.ref_policy_kl_penalty=0.1 \
    " ;;
    *) echo "choose run_stage(sft | lora | dpo)"; exit 1;
    esac

    # 以下为通用执行命令，无特殊可不用修改
    echo "Run with: device_num=${device_num}, run_mode=${run_mode}, run_stage=${run_stage}"
    echo "train_cmd: ${train_cmd}  log_file: ${log_file}"
    export PYTHONPATH=/opt/apex:$PYTHONPATH
    if [ ${run_stage} = "dpo" ];then
        export_metric_effective="${model_name_or_path}-${run_stage}-train/effective_tokens_per_second_per_device"
        export_metric_total="${model_name_or_path}-${run_stage}-train/total_tokens_per_second_per_device "
        timeout 15m python -m torch.distributed.launch \
            --use-env \
            --nproc-per-node=$PADDLE_TRAINER_COUNT \
            --nnodes=$PADDLE_TRAINERS_NUM \
            --node-rank=$PADDLE_TRAINER_ID \
            --master-addr=$POD_0_IP \
            --master-port=$PORT \
            /opt/NeMo-Aligner/examples/nlp/gpt/train_gpt_dpo.py \
            --config-path=/opt/nemo-benchmark/conf_dpo \
            --config-name=${model_name_or_path}.yaml \
            exp_manager.exp_dir=${OUTPUT_DIR} \
            exp_manager.explicit_log_dir=${OUTPUT_DIR} \
            ++exp_manager.log_step_performance=True \
            ++exp_manager.create_mlflow_logger=True \
            ++exp_manager.mlflow_logger_kwargs.experiment_name=${experiment_name} \
            ++exp_manager.mlflow_logger_kwargs.prefix=${experiment_name} \
            trainer.devices=$PADDLE_TRAINER_COUNT \
            trainer.num_nodes=$PADDLE_TRAINERS_NUM \
            ++model.tensor_model_parallel_size=${tensor_model_parallel_size} \
            model.sequence_parallel=False \
            ++model.pipeline_model_parallel_size=${pipeline_model_parallel_size} \
            model.virtual_pipeline_model_parallel_size=1 \
            model.overlap_p2p_comm=False \
            ++model.activations_checkpoint_granularity=${recompute} \
            ++model.activations_checkpoint_method=${activations_checkpoint_method} \
            ++model.activations_checkpoint_num_layers=${recompute_layers} \
            ++model.micro_batch_size=1 \
            ++model.global_batch_size=32 \
            ++model.encoder_seq_length=4096 \
            "model.data.data_prefix={train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${VALID_DATA_PATH}]}" \
            exp_manager.create_wandb_logger=false \
            exp_manager.wandb_logger_kwargs.project=dpo_training \
            exp_manager.wandb_logger_kwargs.name=dpo_training \
            exp_manager.explicit_log_dir=${OUTPUT_DIR} \
            ++trainer.dpo.max_epochs=1 \
            ++trainer.dpo.max_steps=20 \
            ++trainer.dpo.save_interval=1000 \
            ++trainer.dpo.val_check_interval=1000 \
            ++trainer.dpo.limit_val_batches=0.1 \
            ++model.dpo.ref_policy_kl_penalty=0.1 > ${log_file} 2>&1
    else
        export_metric_effective="${model_name_or_path}-${run_stage}-effective_tokens_per_second_per_device"
        export_metric_total="${model_name_or_path}-${run_stage}-total_tokens_per_second_per_device"
        timeout 15m ${train_cmd} > ${log_file} 2>&1
    fi
    
    effective_tokens_per_second_per_device=`find mlruns/ -path */${export_metric_effective}  -print0 | xargs -0 tail -n 1 | awk '{print $2}'`
    echo "effective_tokens_per_second_per_device: ${effective_tokens_per_second_per_device}" >> ${log_file}
    total_tokens_per_second_per_device=`find mlruns/ -path */${export_metric_total}  -print0 | xargs -0 tail -n 1 | awk '{print $2}'`
    echo "total_tokens: ${total_tokens_per_second_per_device}" >> ${log_file}
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