#!/bin/bash

set -x
#nvprof -o timeline_output_medium -f --cpu-profiling off  --profile-from-start off  python  train.py \
#export CUDA_VISIBLE_DEVICES=7
  echo "Usage: "
  echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh speed|mem large|medium|small static|padding /path/to/log"

task=$1
model_type=$2
rnn_type=$3
batch_size=20
run_log_path=${4:-$(pwd)}

devices_str=${CUDA_VISIBLE_DEVICES//,/ }
gpu_devices=($devices_str)
num_gpu_devices=${#gpu_devices[*]}

model_name=padding_${model_type}_${rnn_type}
cpu_num=$(cat /proc/cpuinfo | grep processor | wc -l)
gpu_num=$(nvidia-smi -L|wc -l)
log_file=${run_log_path}/log_${task}_${model_name}_${num_gpu_devices}
train(){
  echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices"
  if [ $task = "speed" ]; then
      sed -i '93c \    config.gpu_options.allow_growth = False' train.py
  elif [ $task = "mem" ]; then
      echo "this task is: "$task
      sed -i '93c \    config.gpu_options.allow_growth = True' train.py
  fi
  python -u train.py \
    --model_type $model_type \
    --rnn_type $rnn_type > ${log_file} 2>&1 &
  train_pid=$!
  sleep 600
  kill -9 $train_pid
}

analysis_times(){
  skip_step=$1
  count_fields=$2
  awk 'BEGIN{count=0}/avg_time:/{
    step_times[count]=$'${count_fields}';
    count+=1;
  }END{
    print "\n================ Benchmark Result ================"
    print "model:", "'${model_type}'"
    if(count>'${skip_step}'){
      step_latency=0
      step_latency_without_step0_avg=0
      step_latency_without_step0_min=step_times['${skip_step}']
      step_latency_without_step0_max=step_times['${skip_step}']
      for(i=0;i<count;++i){
        step_latency+=step_times[i];
        if(i>='${skip_step}'){
          step_latency_without_step0_avg+=step_times[i];
          if(step_times[i]<step_latency_without_step0_min){
            step_latency_without_step0_min=step_times[i];
          }
          if(step_times[i]>step_latency_without_step0_max){
            step_latency_without_step0_max=step_times[i];
          }
        }
      }
      step_latency/=count;
      step_latency_without_step0_avg/=(count-'${skip_step}')
      printf("average latency (including data reading):\n")
      printf("\tAvg: %.3f steps/s\n", step_latency)
      printf("\tFPS: %.3f examples/s\n", "'${batch_size}'"*step_latency)
      printf("average latency (skip '${skip_step}' steps):\n")
      printf("\tAvg: %.3f steps/s\n", step_latency_without_step0_avg)
      printf("\tMin: %.3f steps/s\n", step_latency_without_step0_min)
      printf("\tMax: %.3f steps/s\n", step_latency_without_step0_max)
      printf("\tFPS: %.3f examples/s\n", "'${batch_size}'"*step_latency_without_step0_avg)
      printf("\n")
    }
  }' ${log_file}
}

if [ $1 = 'mem' ]
then
  echo "test for $task"
  gpu_id=`echo $CUDA_VISIBLE_DEVICES |cut -c1`
  nvidia-smi --id=$gpu_id --query-compute-apps=used_memory --format=csv -lms 100 > ${run_log_path}/${task}_${model_name}_${num_gpu_devices}_gpu_use.log 2>&1 &
  gpu_memory_pid=$!
  train
  kill -9 $gpu_memory_pid
  killall -9 nvidia-smi
  awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "MAX_GPU_MEMORY_USE", max}' ${run_log_path}/${task}_${model_name}_${num_gpu_devices}_gpu_use.log
else
  echo "test for $task"
  job_bt=`date '+%Y%m%d%H%M%S'`
  train
  job_et=`date '+%Y%m%d%H%M%S'`
  hostname=`echo $(hostname)|awk -F '.baidu.com' '{print $1}'`
  monquery -n $hostname -i GPU_AVERAGE_UTILIZATION -s $job_bt -e $job_et -d 60 > ${run_log_path}/${task}_${model_name}_${num_gpu_devices}_gpu_avg_utilization
  monquery -n $hostname -i CPU_USER -s $job_bt -e $job_et -d 60 > ${run_log_path}/${task}_${model_name}_${num_gpu_devices}_cpu_use
  awk '{if(NR>1 && $3 >0){time+=$3;count+=1}} END{if(count>0) avg=time/count; else avg=0; printf("AVG_GPU_USE=%.2f\n" ,avg*'${gpu_num}')}' ${run_log_path}/${task}_${model_name}_${num_gpu_devices}_gpu_avg_utilization
  awk '{if(NR>1 && $3 >0){time+=$3;count+=1}} END{if(count>0) avg=time/count; else avg=0; printf("AVG_CPU_USE=%.2f\n" ,avg*'${cpu_num}')}' ${run_log_path}/${task}_${model_name}_${num_gpu_devices}_cpu_use
  analysis_times 0 10
fi
