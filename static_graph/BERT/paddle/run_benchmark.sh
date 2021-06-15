#!bin/bash
set -xe
if [[ $# -lt 3 ]]; then
    echo "running job dict is {1: speed, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1|3|6 base|large fp32|fp16 sp|mp batch_size 1000(max_iter) seqlen128|seqlen512"
    exit
fi

function _set_params(){
    index="$1"
    model_type="$2"
    fp_mode=$3
    run_mode=${4:-"sp"}
    base_batch_size=${5}
    max_iter=${6}
    seq_len=${7}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi

    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}

    mission_name="语义表示"           # 模型所属任务名称，具体可参考scripts/config.ini                                （必填）
    direction_id=1                   # 任务所属方向，0：CV，1：NLP，2：Rec。                                         (必填)
    skip_steps=10                     # 解析日志，有些模型前几个step耗时长，需要跳过                                    (必填)
    keyword="ips:"                 # 解析日志，筛选出数据所在行的关键字                                             (必填)
    ips_unit="sequences/s"                      # 解析日志，按照分隔符分割后形成的数组索引                                        (必填)
    model_mode=-1                     # 解析日志，具体参考scripts/analysis.py.                                      (必填)

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    # if [[ ${index} -eq 6 ]]; then base_batch_size=78; else base_batch_size=32; fi
    if [[ ${fp_mode} = "fp16" ]]; then
        use_amp=True
    elif [[ ${fp_mode} = "fp32" ]]; then
        use_amp=False
    else
        echo "fp_mode should be fp32 or fp16"
        exit 1
    fi
    model_name="bert_${model_type}_${seq_len}_${fp_mode}_bs${base_batch_size}"
    
    if [ ${run_mode} == "sp" ]; then
        batch_size=`expr ${base_batch_size} \* $num_gpu_devices`
    else
        batch_size=${base_batch_size}
    fi
    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/${model_name}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}

}

function _train(){
    echo "model_type: ${model_type}, seq_len: ${seq_len}, fp_mode: ${fp_mode}, batch_size: ${batch_size}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    train_cmd="--max_predictions_per_seq 20
               --learning_rate 1e-4
               --weight_decay 1e-2
               --adam_epsilon 1e-6
               --warmup_steps 10000
               --output_dir ./tmp2/
               --logging_steps 10
               --save_steps 20000
               --max_steps ${max_iter}
               --input_dir=./wikicorpus_en_${seq_len}
               --model_type bert
               --model_name_or_path bert-${model_type}-uncased
               --batch_size ${batch_size}
               --use_amp ${use_amp}"
    case ${run_mode} in
    sp) train_cmd="python -u run_pretrain.py "${train_cmd} ;;
    mp)
        rm -rf ./mylog_${model_name}
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog_${model_name} --gpus=$CUDA_VISIBLE_DEVICES run_pretrain.py "${train_cmd}
        log_parse_file="mylog_${model_name}/workerlog.0" ;;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac

    timeout 15m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
        export job_fail_flag=1
    else
        echo -e "${model_name}, SUCCESS"
        export job_fail_flag=0
    fi
    kill -9 `ps -ef|grep python |awk '{print $2}'`

    if [ $run_mode = "mp" -a -d mylog_${model_name} ]; then
        rm ${log_file}
        cp mylog_${model_name}/workerlog.0 ${log_file}
    fi
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_run
