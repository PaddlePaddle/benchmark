#!/bin/bash

set -xe

#export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
#export CUDA_VISIBLE_DEVICES="0"

export PD_MODELS_ROOT=/work/models

#开启gc
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=1.0

#export FLAGS_cudnn_deterministic=true
#export FLAGS_enable_parallel_graph=1

if [ $# -ne 2 ]; then
  echo "Usage: "
  echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh speed 32"
  exit
fi

task=$1
base_batchsize=$2

devices_str=${CUDA_VISIBLE_DEVICES//,/ }
gpu_devices=($devices_str)
num_gpu_devices=${#gpu_devices[*]}

batch_size=`expr $base_batchsize \* $num_gpu_devices`
num_epochs=2

log_file=${PWD}/log_${task}_bs${batch_size}_${num_gpu_devices}

train(){
  echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
  WORK_ROOT=$PWD
  cd ${PD_MODELS_ROOT}/PaddleCV/image_classification
  python train.py \
     --model=SE_ResNeXt50_32x4d \
     --batch_size=${batch_size} \
     --total_images=1281167 \
     --class_dim=1000 \
     --image_shape=3,224,224 \
     --model_save_dir=output/ \
     --pretrained_model=SE_ResNext50_32x4d_pretrained/ \
     --data_dir=data/ILSVRC2012 \
     --with_mem_opt=False \
     --with_inplace=True \
     --lr_strategy=cosine_decay \
     --lr=0.1 \
     --l2_decay=1.2e-4 \
     --num_epochs=${num_epochs} > ${log_file} 2>&1 &
  train_pid=$!
  sleep 600
  kill -9 $train_pid
  cd ${WORK_ROOT}
}

analysis_times(){
  skip_step=$1
  count_fields=$2
  sed 's/batch\/sec/\ batch\/sec/' ${log_file} | awk 'BEGIN{count=0}/trainbatch/{
    step_times[count]=$'${count_fields}';
    count+=1;
  }END{
    print "\n================ Benchmark Result ================"
    print "num_epochs:", "'${num_epochs}'"
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
      printf("\tAvg: %.3f s/step\n", step_latency)
      printf("\tFPS: %.3f examples/s\n", "'${batch_size}'"/step_latency)
      printf("average latency (skip '${skip_step}' steps):\n")
      printf("\tAvg: %.3f s/step\n", step_latency_without_step0_avg)
      printf("\tMin: %.3f s/step\n", step_latency_without_step0_min)
      printf("\tMax: %.3f s/step\n", step_latency_without_step0_max)
      printf("\tFPS: %.3f examples/s\n", "'${batch_size}'"/step_latency_without_step0_avg)
      printf("\n")
    }
  }'
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
  analysis_times 2 14
fi
