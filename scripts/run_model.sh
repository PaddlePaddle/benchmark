#!/usr/bin/env bash

function _collect_occupancy() {
    if [[ "${BENCHMARK_MONITOR}" = "" ]]; then
        export BENCHMARK_MONITOR=ON
    fi
    if [[ "${BENCHMARK_MONITOR}" = "ON" ]]; then
        hostname=`echo $(hostname)|awk -F '.baidu.com' '{print $1}'`
        gpu_log_file=${log_file}_gpu
        cpu_log_file=${log_file}_cpu
        monquery -n ${hostname} -i GPU_AVERAGE_UTILIZATION -s $job_bt -e $job_et -d 60 > ${gpu_log_file}
        monquery -n ${hostname} -i CPU_USER -s $job_bt -e $job_et -d 60 > ${cpu_log_file}
        cpu_num=$(cat /proc/cpuinfo | grep processor | wc -l)
        gpu_num=$(nvidia-smi -L|wc -l)

        awk '{if(NR>1 && $3 >0){time+=$3;count+=1}} END{if(count>0) avg=time/count; else avg=0; printf("AVG_GPU_USE=%.2f\n" ,avg*'${gpu_num}')}' ${gpu_log_file}
        awk '{if(NR>1 && $3 >0){time+=$3;count+=1}} END{if(count>0) avg=time/count; else avg=0; printf("AVG_CPU_USE=%.2f\n" ,avg*'${cpu_num}')}' ${cpu_log_file}
    fi
}

function _run(){
    # running job dict is {1: speed, 2:mem, 3:profiler, 6:max_batch_size}
    if [[ ${index} -eq 1 ]]; then
        job_bt=`date '+%Y%m%d%H%M%S'`
        _train
        job_et=`date '+%Y%m%d%H%M%S'`
        _collect_occupancy
    elif [[ ${index} -eq 2 ]]; then
        #若测试最大batchsize，FLAGS_fraction_of_gpu_memory_to_use=1
        export FLAGS_fraction_of_gpu_memory_to_use=0.001
        gpu_id=`echo $CUDA_VISIBLE_DEVICES | cut -c1`
        nvidia-smi --id=$gpu_id --query-compute-apps=used_memory --format=csv -lms 100 > gpu_use.log 2>&1 &
        gpu_memory_pid=$!
        _train
        kill ${gpu_memory_pid}
        awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "MAX_GPU_MEMORY_USE=", max}' gpu_use.log
    elif [[ ${index} -eq 3 ]]; then
        job_bt=`date '+%Y%m%d%H%M%S'`
        _train
        job_et=`date '+%Y%m%d%H%M%S'`
    elif [[ ${index} -eq 6 ]]; then
        _train
        error_string="Cannot allocate"
        if [ `grep -c "${error_string}" ${log_parse_file}` -eq 0 ]; then
            echo "MAX_BATCH_SIZE=${batch_size}"
        else
            echo "MAX_BATCH_SIZE=0"
        fi
    else
        echo "Do nothing"
    fi

    python ${BENCHMARK_ROOT}/scripts/analysis.py \
            --filename "${log_file}" \
            --log_with_profiler ${log_with_profiler:-"not found!"} \
            --profiler_path ${profiler_path:-"not found!"} \
            --keyword "${keyword}" \
            --separator "${separator}" \
            --position ${position} \
            --base_batch_size ${base_batch_size} \
            --skip_steps ${skip_steps} \
            --model_mode ${model_mode} \
            --model_name "${model_name}" \
            --run_mode ${run_mode} \
            --index ${index} \
            --gpu_num ${num_gpu_devices} \
            --range ${range:-""}
}

