#!/bin/bash
# Occupy all GPU memory (5% reserved actually)
export FLAGS_fraction_of_gpu_memory_to_use=1.0

# Enable gc when this flag is larger than or equal to 0. 
# If you change this value, please make sure that the large
# model can run when batch_size = 2100; and the small model
# can run when batch_size = 5365
export FLAGS_eager_delete_tensor_gb=0.0

# You can set this ratio to control the number of gc ops 
# GC is disabled when this flag is 0; and full gc would be
# performed when this flag is 1. Must be inside [0, 1].
# If you change this value, please make sure that the large
# model can run when batch_size = 2100; and the small model
# can run when batch_size = 5365
export FLAGS_memory_fraction_of_eager_deletion=1.0

if [ $# -ne 3 ]; then
  echo "Usage: "
  echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh speed|mem large|small 20"
  exit
fi

task=$1
model_type=$2
base_batchsize=$3

devices_str=${CUDA_VISIBLE_DEVICES//,/ }
gpu_devices=($devices_str)
num_gpu_devices=${#gpu_devices[*]}

batch_size=`expr $base_batchsize \* $num_gpu_devices`

log_file=log_${model_type}_${task}_${batch_size}_${num_gpu_devices}

train(){
  echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
  python train.py --data_path data/simple-examples/data/ \
    --model_type $model_type \
    --use_gpu True \
    --enable_ce \
    --batch_size $batch_size > ${log_file} 2>&1 &
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
    print "batch_size:", "'${batch_size}'"
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
  export FLAGS_fraction_of_gpu_memory_to_use=0.001
  gpu_id=`echo $CUDA_VISIBLE_DEVICES |cut -c1`
  nvidia-smi --id=$gpu_id --query-compute-apps=used_memory --format=csv -lms 100 > gpu_use.log 2>&1 &
  gpu_memory_pid=$!
  train
  kill $gpu_memory_pid
  awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "Max=", max}' gpu_use.log
else
  echo "test for $task"
  train
  analysis_times 0 9
fi