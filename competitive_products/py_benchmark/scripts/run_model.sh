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
        gpu_id=`echo $CUDA_VISIBLE_DEVICES | cut -c1`
        nvidia-smi --id=$gpu_id --query-compute-apps=used_memory --format=csv -lms 100 > ${log_file}_gpu_use.log 2>&1 &            #for mem max
        gpu_memory_pid=$!                                                                                              #for mem max
        job_bt=`date '+%Y%m%d%H%M%S'`
        _train
        job_et=`date '+%Y%m%d%H%M%S'`
        _collect_occupancy                                                                                             #for cpu rate and gpu rate
        kill ${gpu_memory_pid}
        killall -9 nvidia-smi
        #awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "MAX_GPU_MEMORY_USE=", max}' ${log_file}_gpu_use.log      #for mem max 
        cat ${log_file}_gpu_use.log | tr -d ' MiB' | awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "MAX_GPU_MEMORY_USE=", max}'
#######################################################################

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

