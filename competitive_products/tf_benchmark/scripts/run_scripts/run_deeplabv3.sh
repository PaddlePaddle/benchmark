#!/bin/bash

set -x
echo " CUDA_VISIBLE_DEVICES=7 bash run_deeplabv3.sh speed sp /log/file"
#export CUDA_VISIBLE_DEVICES="0,1,2,3"
#export CUDA_VISIBLE_DEVICES="0"

export TF_MODELS_ROOT=/ssd3/heya/tensorflow/tensorflow_models/models/research # NOTE: this is in 19 local

device=${CUDA_VISIBLE_DEVICES//,/ }
arr=($device)
num_gpu_devices=${#arr[*]}

train_crop_size=513
total_step=80
base_batch_size=2
batch_size=`expr ${base_batch_size} \* ${num_gpu_devices}`

task=${1}
run_mode=${2}
run_log_path=${3:-$(pwd)}
model_name=deeplabv3

log_file=${run_log_path}/log_${model_name}_${task}_${num_gpu_devices}

train(){
  echo "Train on ${num_gpu_devices} GPUs"
  export PYTHONPATH=${TF_MODELS_ROOT}:${TF_MODELS_ROOT}/slim
  if [ ${task} = "mem" ]; then
      sed -i '369c \    session_config.gpu_options.allow_growth = True' ${TF_MODELS_ROOT}/deeplab/train.py
  elif [ ${task} = "speed" ]; then
      sed -i '369c \    session_config.gpu_options.allow_growth = False' ${TF_MODELS_ROOT}/deeplab/train.py
  fi
  rm -rf ./train_logs

  python ${TF_MODELS_ROOT}/deeplab/train.py  \
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
           --dataset_dir=/ssd1/ljh/dataset/tf_cityscapes/tfrecord/  \
           --tf_initial_checkpoint=${TF_MODELS_ROOT}/deeplab/deeplabv3_cityscapes_train/model.ckpt.index  \
           --num_clones ${num_gpu_devices} > ${log_file} 2>&1 &
  train_pid=$!
  sleep 400
  kill =9 $train_pid
}

analysis_times(){
  awk 'BEGIN{count=0}/global\ step/{
    split($0, b, "(");
    a=b[2];
    step_times[count]=a;
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
      printf("\tFPS: %.3f images/s\n", "'${num_gpu_devices}'"*"'${batch_size}'"/step_latency)
      printf("average latency (including data reading, without step 0):\n")
      printf("\tAvg: %.3f s/step\n", step_latency_without_step0_avg)
      printf("\tMin: %.3f s/step\n", step_latency_without_step0_min)
      printf("\tMax: %.3f s/step\n", step_latency_without_step0_max)
      printf("\tFPS: %.3f images/s\n", "'${num_gpu_devices}'"*"'${batch_size}'"/step_latency_without_step0_avg)
      printf("\n")
    }
  }' ${log_file}
}

echo "test for $1"

gpu_id=`echo $CUDA_VISIBLE_DEVICES | cut -c1`
nvidia-smi --id=$gpu_id --query-compute-apps=used_memory --format=csv -lms 100 > ${log_file}_gpu_use.log 2>&1 &            #for mem max
gpu_memory_pid=$!

job_bt=`date '+%Y%m%d%H%M%S'`
train
job_et=`date '+%Y%m%d%H%M%S'`

hostname=`echo $(hostname)|awk -F '.baidu.com' '{print $1}'`
monquery -n $hostname -i GPU_AVERAGE_UTILIZATION -s $job_bt -e $job_et -d 60 > ${log_file}_gpu_avg_utilization
monquery -n $hostname -i CPU_USER -s $job_bt -e $job_et -d 60 > ${log_file}_cpu_use
cpu_num=$(cat /proc/cpuinfo | grep processor | wc -l)
gpu_num=$(nvidia-smi -L|wc -l)
awk '{if(NR>1 && $3 >0){time+=$3;count+=1}} END{if(count>0) avg=time/count; else avg=0; printf("AVG_GPU_USE=%.2f\n" ,avg*'${gpu_num}')}' ${log_file}_gpu_avg_utilization
awk '{if(NR>1 && $3 >0){time+=$3;count+=1}} END{if(count>0) avg=time/count; else avg=0; printf("AVG_CPU_USE=%.2f\n" ,avg*'${cpu_num}')}' ${log_file}_cpu_use

kill ${gpu_memory_pid}
killall -9 nvidia-smi
cat ${log_file}_gpu_use.log | tr -d ' MiB' | awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "MAX_GPU_MEMORY_USE=", max}'

analysis_times
