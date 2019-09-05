#!bin/bash
set -xe

if [[ $# -lt 3 ]]; then
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh speed|mem|maxbs base|big fp32|fp16 sp|mp /ssd1/ljh/logs"
    exit
fi

function _set_params(){
    index="$1"
    model_mode="$2"
    fp_mode=$3
    run_mode=${4:-"sp"}
    run_log_path=${5:-$(pwd)}

    model_name="bert_${model_mode}_${fp_mode}"
    skip_steps=1
    keyword="speed:"
    separator=" "
    position=13
    model_mode=1

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    if [[ ${index} = "maxbs" ]]; then base_batch_size=78; else base_batch_size=32; fi
    batch_size=`expr ${base_batch_size} \* $num_gpu_devices`
    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_parse_file=${log_file}

}

function _set_env(){
    export FLAGS_cudnn_deterministic=true
    export FLAGS_enable_parallel_graph=0
    export FLAGS_eager_delete_tensor_gb=0.0
    export FLAGS_fraction_of_gpu_memory_to_use=1.0
}

function _train(){
    BERT_BASE_PATH=$(pwd)/chinese_L-12_H-768_A-12
    TASK_NAME='XNLI'
    DATA_PATH=$(pwd)/data
    CKPT_PATH=$(pwd)/save
    train_cmd=" --task_name ${TASK_NAME} \
          --use_cuda true \
          --do_train true \
          --do_val true \
          --do_test true \
          --batch_size ${base_batch_size} \
          --in_tokens False \
          --init_pretraining_params ${BERT_BASE_PATH}/params \
          --data_dir ${DATA_PATH} \
          --vocab_path ${BERT_BASE_PATH}/vocab.txt \
          --checkpoints ${CKPT_PATH} \
          --save_steps 1000 \
          --weight_decay  0.01 \
          --warmup_proportion 0.1 \
          --validation_steps 1000 \
          --epoch 2 \
          --max_seq_len 128 \
          --bert_config_path ${BERT_BASE_PATH}/bert_config.json \
          --learning_rate 5e-5 \
          --skip_steps 100 \
          --random_seed 1"

    case ${run_mode} in
    sp) train_cmd="python -u run_classifier.py "${train_cmd} ;;
    mp)
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --selected_gpus=$CUDA_VISIBLE_DEVICES run_classifier.py "${train_cmd}
        log_parse_file="mylog/workerlog.0" ;;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac

    if [[ ${fp_mode} == "fp16" ]]; then
        train_cmd=${train_cmd}" --use_fp16=true --loss_scaling=8.0"
    fi

    ${train_cmd} > ${log_file} 2>&1 &
    train_pid=$!
    sleep 600
    kill -9 `ps -ef|grep python |awk '{print $2}'`

    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_set_env
_run