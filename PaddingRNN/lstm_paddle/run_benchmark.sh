#!bin/bash
set -xe

if [[ $# -lt 3 ]]; then
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh speed|mem large|medium|small static|padding sp|mp /ssd1/ljh/logs"
    exit
fi

function _set_params(){
    index=$1
    model_type=$2
    rnn_type=$3
    run_mode=${4:-"sp"}
    run_log_path=${5:-$(pwd)}

    model_name="paddingrnn_"${model_type}_${rnn_type}
    skip_steps=0
    keyword="avg_time:"
    separator=" "
    position=8
    model_mode=1

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    if [[ ${index} = "maxbs" ]]; then base_batch_size=12000; else base_batch_size=20; fi
    batch_size=`expr ${base_batch_size} \* $num_gpu_devices`
    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_parse_file=${log_file}

}

function _set_env(){
    export MKL_NUM_THREADS=1
    export OMP_NUM_THREADS=1

    # Occupy all GPU memory (5% reserved actually)
    export FLAGS_fraction_of_gpu_memory_to_use=1.0
    export FLAGS_eager_delete_tensor_gb=0.0
    export FLAGS_memory_fraction_of_eager_deletion=0.5
    if [[ ${model_type} == "large" ]]; then
        export FLAGS_memory_fraction_of_eager_deletion=1.0
    fi
}

function _train(){
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
    python train.py --data_path data/simple-examples/data/ \
      --model_type $model_type \
      --use_gpu True \
      --enable_ce \
      --max_epoch=5 \
      --rnn_model ${rnn_type} \
      --use_py_reader True \
      --batch_size ${batch_size} > ${log_file} 2>&1
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