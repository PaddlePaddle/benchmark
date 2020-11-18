#!bin/bash
set -xe

if [[ $# -lt 3 ]]; then
    echo "running job dict is {1: speed, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1|3|6 base|large fp32|fp16 sp|mp 1000(max_iter)"
    exit
fi

function _set_params(){
    index="$1"
    model_type="$2"
    fp_mode=$3
    run_mode=${4:-"sp"}
    max_iter=${5}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi

    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}

    model_name="bert_${model_type}_${fp_mode}"
    mission_name="语义表示"           # 模型所属任务名称，具体可参考scripts/config.ini                                （必填）
    direction_id=1                   # 任务所属方向，0：CV，1：NLP，2：Rec。                                         (必填)
    skip_steps=1                     # 解析日志，有些模型前几个step耗时长，需要跳过                                    (必填)
    keyword="speed:"                 # 解析日志，筛选出数据所在行的关键字                                             (必填)
    separator=" "                    # 解析日志，数据所在行的分隔符                                                  (必填)
    position=13                      # 解析日志，按照分隔符分割后形成的数组索引                                        (必填)
    model_mode=1                     # 解析日志，具体参考scripts/analysis.py.                                      (必填)

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    # if [[ ${index} -eq 6 ]]; then base_batch_size=78; else base_batch_size=32; fi
    if [[ ${model_type} = "base" ]]; then base_batch_size=32; else base_batch_size=8; fi
    batch_size=`expr ${base_batch_size} \* $num_gpu_devices`
    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/${model_name}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}

}

function _set_env(){
    export FLAGS_cudnn_deterministic=true
    export FLAGS_enable_parallel_graph=0
    export FLAGS_eager_delete_tensor_gb=0.0
    export FLAGS_fraction_of_gpu_memory_to_use=1.0
}

function _train(){
    if [[ ${model_type} = "base" ]]; then
        BERT_BASE_PATH=$(pwd)/chinese_L-12_H-768_A-12
        TASK_NAME='XNLI'
        DATA_PATH=$(pwd)/data
    elif [[ ${model_type} = "large" ]]; then
        BERT_BASE_PATH=$(pwd)/uncased_L-24_H-1024_A-16
        TASK_NAME='mnli'
        DATA_PATH=$(pwd)/MNLI
    fi
    CKPT_PATH=$(pwd)/save
    train_cmd=" --task_name ${TASK_NAME} \
          --use_cuda true \
          --do_train true \
          --do_val False \
          --do_test False \
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
          --is_profiler=${is_profiler} \
          --profiler_path=${profiler_path} \
          --max_iter=${max_iter} \
          --max_seq_len 128 \
          --bert_config_path ${BERT_BASE_PATH}/bert_config.json \
          --learning_rate 5e-5 \
          --skip_steps 100 \
          --random_seed 1"

    case ${run_mode} in
    sp) train_cmd="python -u run_classifier.py "${train_cmd} ;;
    mp)
        rm -rf ./mylog
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --selected_gpus=$CUDA_VISIBLE_DEVICES run_classifier.py "${train_cmd}
        log_parse_file="mylog/workerlog.0" ;;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac

    if [[ ${fp_mode} == "fp16" ]]; then
        train_cmd=${train_cmd}" --use_fp16=true "
    fi

    timeout 15m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
        export job_fail_flag=1
    else
        echo -e "${model_name}, SUCCESS"
        export job_fail_flag=0
    fi
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
