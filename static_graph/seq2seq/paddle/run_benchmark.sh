#!bin/bash
set -xe

if [[ $# -lt 4 ]]; then
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh speed|mem|maxbs sp|mp /ssd1/ljh/logs"
    exit
fi

function _set_params(){
    index=$1
    base_batch_size=128
    model_name="seq2seq"

    run_mode=${2:-"sp"}
    run_log_path=${3:-$(pwd)}

    skip_steps=2
    keyword="trainbatch"
    separator=" "
    position=-1
    model_mode=1

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_parse_file=${log_file}

}

function _set_env(){
    #开启gc
    echo "nothing"
}

function _train(){
   python train.py \
          --src_lang en --tar_lang vi \
          --attention True \
          --num_layers 2 \
          --hidden_size 512 \
          --src_vocab_size 17191 \
          --tar_vocab_size 7709 \
          --batch_size ${batch_size} \
          --dropout 0.2 \
          --init_scale  0.1 \
          --max_grad_norm 5.0 \
          --train_data_prefix data/en-vi/train \
          --eval_data_prefix data/en-vi/tst2012 \
          --test_data_prefix data/en-vi/tst2013 \
          --vocab_prefix data/en-vi/vocab \
          --use_gpu True \
          --max_epoch 2  > ${log_file} 2>&1
}

function _run(){
    if [[ ${index} = "mem" ]]; then
        #若测试最大batchsize，FLAGS_fraction_of_gpu_memory_to_use=1
        export FLAGS_fraction_of_gpu_memory_to_use=0.001
        gpu_id=`echo $CUDA_VISIBLE_DEVICES | cut -c1`
        nvidia-smi --id=$gpu_id --query-compute-apps=used_memory --format=csv -lms 100 > gpu_use.log 2>&1 &
        gpu_memory_pid=$!
        _train
        kill ${gpu_memory_pid}
        awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "MAX_GPU_MEMORY_USE=", max}' gpu_use.log
    elif [[ ${index} = "speed" ]]; then
        job_bt=`date '+%Y%m%d%H%M%S'`
        _train
        job_et=`date '+%Y%m%d%H%M%S'`
        hostname=`echo $(hostname)|awk -F '.baidu.com' '{print $1}'`
        gpu_log_file=${log_file}_gpu
        cpu_log_file=${log_file}_cpu
        monquery -n ${hostname} -i GPU_AVERAGE_UTILIZATION -s $job_bt -e $job_et -d 60 > ${gpu_log_file}
        monquery -n ${hostname} -i CPU_USER -s $job_bt -e $job_et -d 60 > ${cpu_log_file}
        cpu_num=$(cat /proc/cpuinfo | grep processor | wc -l)
        gpu_num=$(nvidia-smi -L|wc -l)

        awk '{if(NR>1 && $3 >0){time+=$3;count+=1}} END{if(count>0) avg=time/count; else avg=0; printf("AVG_GPU_USE=%.2f\n" ,avg*'${gpu_num}')}' ${gpu_log_file}
        awk '{if(NR>1 && $3 >0){time+=$3;count+=1}} END{if(count>0) avg=time/count; else avg=0; printf("AVG_CPU_USE=%.2f\n" ,avg*'${cpu_num}')}' ${cpu_log_file}
    else
        _train
        error_string="Please shrink FLAGS_fraction_of_gpu_memory_to_use"
        if [ `grep -c "${error_string}" ${log_parse_file}` -eq 0 ]; then
            echo "MAX_BATCH_SIZE=${batch_size}"
        else
            echo "MAX_BATCH_SIZE=0"
        fi
    fi

    python ${BENCHMARK_ROOT}/utils/analysis.py \
            --filename ${log_file} \
            --keyword ${keyword} \
            --separator "${separator}" \
            --position ${position} \
            --base_batch_size ${base_batch_size} \
            --skip_steps ${skip_steps} \
            --model_mode ${model_mode} \
            --model_name ${model_name} \
            --run_mode ${run_mode} \
            --index ${index} \
            --gpu_num ${num_gpu_devices}
}

_set_params $@
_set_env
_run