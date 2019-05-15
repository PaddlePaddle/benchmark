#!bin/bash
set -xe

#export FLAGS_cudnn_deterministic=true
#export FLAGS_enable_parallel_graph=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98
export FLAGS_memory_fraction_of_eager_deletion=1.0
export FLAGS_conv_workspace_size_limit=1500

if [ $# -lt 2 ]; then
  echo "Usage: "
  echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh train|infer speed|mem|maxbs /ssd1/ljh/logs"
  exit
fi

task="$1"
index="$2"
run_log_path=${3:-$(pwd)}
model_name="mask_rcnn"

device=${CUDA_VISIBLE_DEVICES//,/ }
arr=($device)
num_gpu_devices=${#arr[*]}
if [ $index = "maxbs" ]; then base_batch_size=5; else base_batch_size=1; fi
batch_size=`expr ${base_batch_size} \* $num_gpu_devices`
log_file=${run_log_path}/${model_name}_${task}_${index}_${num_gpu_devices}
# The default learning_rate is 0.01, which is used for training with 8 GPUs.
learning_rate=$(awk 'BEGIN{ print 0.00125 * '${num_gpu_devices}' }')

train(){
  echo "Train on ${num_gpu_devices} GPUs"
  echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
  python train.py \
   --model_save_dir=output/ \
   --pretrained_model=../imagenet_resnet50_fusebn/ \
   --data_dir=./dataset/coco \
   --im_per_batch=${base_batch_size} \
   --learning_rate=${learning_rate} \
   --MASK_ON=True > ${log_file} 2>&1 &
  train_pid=$!
  sleep 600
  kill -9 $train_pid
}

infer(){
  echo "infer on ${num_gpu_devices} GPUs"
  echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
  python eval_coco_map.py \
    --dataset=coco2017 \
    --pretrained_model=../imagenet_resnet50_fusebn/ \
    --MASK_ON=True > ${log_file} 2>&1 &
  infer_pid=$!
  sleep 60
  kill -9 $infer_pid
}

analysis_times(){
  skip_step=$1
  filter_fields=$2
  count_fields=$3
  awk 'BEGIN{count=0}{if(NF=='${filter_fields}'){
    step_times[count]=$'${count_fields}';
    count+=1;}
  }END{
    print "\n================ Benchmark Result ================"
    print "total_step:", count
    print "batch_size:", "'${batch_size}'"
    if(count>1){
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
      printf("average latency (origin result):\n")
      printf("\tAvg: %.3f s/step\n", step_latency)
      printf("\tFPS: %.3f examples/s\n", "'${batch_size}'"/step_latency)
      printf("average latency (skip '${skip_step}' steps):\n")
      printf("\tAvg: %.3f s/step\n", step_latency_without_step0_avg)
      printf("\tMin: %.3f s/step\n", step_latency_without_step0_min)
      printf("\tMax: %.3f s/step\n", step_latency_without_step0_max)
      printf("\tFPS: %.3f examples/s\n", '${batch_size}'/step_latency_without_step0_avg)
      printf("\n")
    }
  }' ${log_file}
}

echo "Benchmark for $index"

if [ $index = "mem" ]
then
  export FLAGS_fraction_of_gpu_memory_to_use=0.001
  gpu_id=`echo $CUDA_VISIBLE_DEVICES | cut -c1`
  nvidia-smi --id=$gpu_id --query-compute-apps=used_memory --format=csv -lms 100 > gpu_use.log 2>&1 &
  gpu_memory_pid=$!
  $task
  kill $gpu_memory_pid
  awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "Max=", max}' gpu_use.log
elif [ ${index} = 'speed' ]
then
  $task
  if [ ${task} = "train" ]
  then
      analysis_times 3 20 20
  else
      analysis_times 3 5 5
  fi
else
  $task
  error_string="Please shrink FLAGS_fraction_of_gpu_memory_to_use or FLAGS_initial_gpu_memory_in_mb or FLAGS_reallocate_gpu_memory_in_mbenvironment variable to a lower value"
  if [ `grep -c "${error_string}" ${log_file}` -eq 0 ]; then
    echo "maxbs is ${batch_size}"
  else
    echo "maxbs running error"
  fi
fi
