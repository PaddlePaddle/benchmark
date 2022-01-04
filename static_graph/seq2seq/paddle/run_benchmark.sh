#!bin/bash
set -xe

if [[ $# -lt 1 ]]; then
    echo "running job dict is {1: speed, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1|3|6 sp|mp 1(max_epoch)"
    exit
fi

function _set_params(){
    index=$1
    base_batch_size=128
    model_name="seq2seq"_bs${base_batch_size}

    run_mode="sp" # Don't support mp
    max_epoch=${3}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi
 
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}

    mission_name="文本生成"           # 模型所属任务名称，具体可参考scripts/config.ini                                （必填）
    direction_id=1                   # 任务所属方向，0：CV，1：NLP，2：Rec。                                         (必填)
    skip_steps=0                     # 解析日志，有些模型前几个step耗时长，需要跳过                                    (必填)
    keyword="ips:"              # 解析日志，筛选出数据所在行的关键字                                             (必填)
    model_mode=-1                     # 解析日志，s/step -> steps/s 具体参考scripts/analysis.py.                    (必填)
    ips_unit="tokens/s"

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}
    if [ ${num_gpu_devices} -gt 1 ]; then
        echo "Multi-GPU training is not supported yet."
        exit
    fi

    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/${model_name}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}
}

function _set_env(){
    #开启gc
    echo "nothing"
}

function _train(){
   train_cmd="--src_lang en --tar_lang vi \
              --attention True \
              --num_layers 2 \
              --hidden_size 512 \
              --src_vocab_size 17191 \
              --tar_vocab_size 7709 \
              --batch_size ${base_batch_size} \
              --dropout 0.2 \
              --init_scale  0.1 \
              --max_grad_norm 5.0 \
              --train_data_prefix data/en-vi/train \
              --eval_data_prefix data/en-vi/tst2012 \
              --test_data_prefix data/en-vi/tst2013 \
              --vocab_prefix data/en-vi/vocab \
              --use_gpu True \
              --profiler_path=${profiler_path} \
              --max_epoch=${max_epoch}"
    
    if [[ ${is_profiler} -eq 1 ]]; then
        timeout 15m python -u train.py \
               --profile \
               ${train_cmd} > ${log_file} 2>&1
    elif [[ ${is_profiler} -eq 0 ]]; then
        timeout 15m python -u train.py ${train_cmd} > ${log_file} 2>&1
        if [ $? -ne 0 ];then
            echo -e "${model_name}, FAIL"
            export job_fail_flag=1
        else
            echo -e "${model_name}, SUCCESS"
            export job_fail_flag=0
        fi
    fi
    kill -9 `ps -ef|grep python |awk '{print $2}'`
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_set_env
_run
