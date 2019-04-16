#!/bin/bash

set -xe

export CUDA_VISIBLE_DEVICES="0,1,2,3"
#export CUDA_VISIBLE_DEVICES="1"

export TF_MODELS_ROOT=/work/tensorflow/models

gpu_devices=(`echo ${CUDA_VISIBLE_DEVICES} | tr "," " "`)
num_gpu_devices=${#gpu_devices[@]}

train_crop_size=513
total_step=80
batch_size=$((2 * num_gpu_devices))

task=speed
log_file=log_${task}_bs${batch_size}_${num_gpu_devices}

train(){
  echo "Train on ${num_gpu_devices} GPUs"
  export PYTHONPATH=${TF_MODELS_ROOT}/research:${TF_MODELS_ROOT}/research/slim
  python ${TF_MODELS_ROOT}/research/deeplab/train.py  \
           --logtostderr  \
           --training_number_of_steps=${total_step}  \
           --train_split=train  \
           --model_variant=xception_65  \
           --atrous_rates=6  \
           --atrous_rates=12  \
           --atrous_rates=18  \
           --output_stride=16  \
           --decoder_output_stride=4  \
           --train_crop_size=${train_crop_size}  \
           --train_crop_size=${train_crop_size}  \
           --train_batch_size=${batch_size}  \
           --dataset=cityscapes  \
           --train_logdir=./train_logs  \
           --dataset_dir=${TF_MODELS_ROOT}/research/deeplab/datasets/cityscapes/tfrecord/  \
           --tf_initial_checkpoint=./deeplabv3_cityscapes_train/model.ckpt.index  \
           --num_clones ${num_gpu_devices} > ${log_file} 2>&1
}

analysis_times(){
  awk 'BEGIN{count=0}/global_step\/sec:/{
    step_times[count]=1/$2;
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
      printf("\tFPS: %.3f images/s\n", "'${batch_size}'"/step_latency)
      printf("average latency (including data reading, without step 0):\n")
      printf("\tAvg: %.3f s/step\n", step_latency_without_step0_avg)
      printf("\tMin: %.3f s/step\n", step_latency_without_step0_min)
      printf("\tMax: %.3f s/step\n", step_latency_without_step0_max)
      printf("\tFPS: %.3f images/s\n", "'${batch_size}'"/step_latency_without_step0_avg)
      printf("\n")
    }
  }' ${log_file} 
}

echo "test for $1"
train
analysis_times
