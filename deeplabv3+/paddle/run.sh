#!/bin/bash

set -xe

#export FLAGS_cudnn_deterministic=true
#export FLAGS_enable_parallel_graph=1

export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fast_eager_deletion_mode=1

if [ $# -ne 1 ]; then
  echo "Usage: "
  echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh speed|mem"
  exit
fi

task="$1"

DATASET_PATH=${PWD}/data/cityscape/
INIT_WEIGHTS_PATH=${PWD}/deeplabv3plus_xception65_initialize
SAVE_WEIGHTS_PATH=${PWD}/output/model
echo $DATASET_PATH

devices_str=${CUDA_VISIBLE_DEVICES//,/ }
gpu_devices=($devices_str)
num_gpu_devices=${#gpu_devices[*]}

train_crop_size=513
total_step=80
batch_size=`expr 2 \* $num_gpu_devices`

log_file=log_${task}_bs${batch_size}_${num_gpu_devices}

train(){
  echo "Train on ${num_gpu_devices} GPUs"
  echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
  python $PWD/train.py \
      --batch_size=${batch_size} \
      --train_crop_size=${train_crop_size} \
      --total_step=${total_step} \
      --init_weights_path=${INIT_WEIGHTS_PATH} \
      --save_weights_path=${SAVE_WEIGHTS_PATH} \
      --dataset_path=${DATASET_PATH} \
      --parallel=True > ${log_file} 2>&1

  # Python multi-processing is used to read images, so need to
  # kill those processes if the main train process is aborted.
  #ps -aux | grep "$PWD/train.py" | awk '{print $2}' | xargs kill -9
}

analysis_times(){
  awk 'BEGIN{count=0}/step_time_cost:/{
    step_times[count]=$6;
    count+=1;
  }END{
    print "\n================ Benchmark Result ================"
    print "total_step:", "'${total_step}'"
    print "batch_size:", "'${batch_size}'"
    print "train_crop_size:", "'${train_crop_size}'"
    if(count>1){
      step_latency=0
      step_latency_without_step0_avg=0
      step_latency_without_step0_min=step_times[1]
      step_latency_without_step0_max=step_times[1]
      for(i=0;i<count;++i){
        step_latency+=step_times[i];
        if(i>0){
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
      step_latency_without_step0_avg/=(count-1)
      printf("average latency (including data reading):\n")
      printf("\tAvg: %.3f s/step\n", step_latency)
      printf("\tFPS: %.3f examples/s\n", "'${batch_size}'"/step_latency)
      printf("average latency (including data reading, without step 0):\n")
      printf("\tAvg: %.3f s/step\n", step_latency_without_step0_avg)
      printf("\tMin: %.3f s/step\n", step_latency_without_step0_min)
      printf("\tMax: %.3f s/step\n", step_latency_without_step0_max)
      printf("\tFPS: %.3f examples/s\n", "'${batch_size}'"/step_latency_without_step0_avg)
      printf("\n")
    }
  }' ${log_file} 
}

if [ $task = "mem" ]
then
  echo "Benchmark for $task"
  export FLAGS_fraction_of_gpu_memory_to_use=0.001
  gpu_id=`echo $CUDA_VISIBLE_DEVICES | cut -c1`
  nvidia-smi --id=$gpu_id --query-compute-apps=used_memory --format=csv -lms 100 > gpu_use.log 2>&1 &
  gpu_memory_pid=$!
  train
  kill $gpu_memory_pid
  awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "Max=", max}' gpu_use.log
else
  echo "Benchmark for $task"
  train
  analysis_times
fi
