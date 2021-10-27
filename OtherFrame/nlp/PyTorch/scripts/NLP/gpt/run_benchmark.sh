#!/usr/bin/env bash
set -xe
# 参数说明
function _set_params(){
    run_mode=${1:-"sp"}
    batch_size=${2:-"64"}
    fp_item=${3:-"fp32"}        # fp32|fp16
    max_iter=${4}       # 如果需要修改代码提前中断
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}

    model_name="nlp_gpt"
    mission_name="语义表示"            # 模型所属任务名称，具体可参考scripts/config.ini  必填）
    direction_id=1                   # 任务所属方向，0：CV，1：NLP，2：Rec。     (必填)
    ips_unit="tokens/s"

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    log_file=${run_log_path}/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}
    index_log_file=${run_log_path}/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}_speed

    CHECKPOINT_PATH=${run_log_path}/results/checkpoints
    DATA_PATH=${run_log_path}/data/my-gpt2_text_document
    TOKEN_FILES=${run_log_path}/token_files

    export PATH=/workspace/run_env:${PATH}
    echo `python3-config --help`
    # rm /usr/local/bin/python3-config  
    # ln -s /usr/local/python3.7.0/bin/python3.7m-config /usr/local/bin/python3-config  
    #export PAPH=$PATH;
}
function _train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
    # 防止checkpoints冲突
    if [ -d $CHECKPOINT_PATH ]; then
        rm -rf $CHECKPOINT_PATH
    fi
    # 如需开启特殊优化flag、参数请注明

    if [ $fp_item = "fp16" ]; then
        use_fp16_cmd="--fp16"
    fi 

    train_cmd="\
       --num-layers 12 \
       --hidden-size 768\
       --num-attention-heads 12\
       --micro-batch-size $batch_size \
       --global-batch-size $(($batch_size*$num_gpu_devices)) \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters $max_iter\
       --lr-decay-iters 320000\
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $TOKEN_FILES/gpt-en-vocab.json \
       --merge-file $TOKEN_FILES/gpt-en-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 1 \
       --save-interval 2000 \
       --eval-interval 500 \
       --eval-iters 10 \
       $use_fp16_cmd
    "
    DISTRIBUTED_ARGS="--nproc_per_node $num_gpu_devices --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6010"

    case ${run_mode} in
    sp) train_cmd=" python -u pretrain_gpt.py "${train_cmd} ;;
    mp)
        train_cmd="python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_gpt.py "${train_cmd} ;;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac

    cd Megatron-LM
    timeout 15m ${train_cmd} > ${log_file} 2>&1
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
    cd -
}
function _analysis_log(){
    #python analysis_log.py ${log_file} ${index_log_file}   # 分析log产出待入库的json 文件
    python analysis.py --filename ${log_file} \
        --keyword "(ms):" \
        --base_batch_size $(($batch_size*1024)) \
        --skip_steps 20 \
        --model_mode -2 \
        --model_name ${model_name}\
        --mission_name ${mission_name}\
        --direction_id ${direction_id} \
        --run_mode ${run_mode} \
        --gpu_num ${num_gpu_devices} \
        --ips_unit ${ips_unit} \
        --time_unit 'ms' > ${index_log_file}  
}

_set_params $@
_train
_analysis_log

