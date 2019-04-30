#!bin/bash
set -xe

#export FLAGS_cudnn_deterministic=true
#export FLAGS_enable_parallel_graph=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98
export FLAGS_memory_fraction_of_eager_deletion=1.0

if [ $# -ne 2 ]; then
  echo "Usage: "
  echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh train|infer speed|mem"
  exit
fi

task="$1"
index="$2"

device=${CUDA_VISIBLE_DEVICES//,/ }
arr=($device)
num_gpu_devices=${#arr[*]}
batch_size=`expr 1 \* $num_gpu_devices`
log_file=log_${task}_${index}_${num_gpu_devices}

train(){
  echo "Train on ${num_gpu_devices} GPUs"
  echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
  if [ ${num_gpu_devices} -eq 1 ]
  then
      python tools/train_net.py \
      --config-file "configs/e2e_mask_rcnn_R_50_C4_1x.yaml" \
      SOLVER.IMS_PER_BATCH 1 \
      SOLVER.BASE_LR 0.0025 \
      SOLVER.MAX_ITER 720000 \
      SOLVER.STEPS "(480000, 640000)" \
      TEST.IMS_PER_BATCH 1 > ${log_file} 2>&1 &
  else
      python -m torch.distributed.launch \
      --nproc_per_node=8 \
      ./tools/train_net.py \
      --config-file "configs/e2e_mask_rcnn_R_50_C4_1x.yaml"\
       SOLVER.IMS_PER_BATCH 8 > ${log_file} 2>&1 &
  fi
  train_pid=$!
  sleep 600
  kill -9 $train_pid
}

infer(){
  echo "infer on ${num_gpu_devices} GPUs"
  echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
#  python eval_coco_map.py \
#    --dataset=coco2017 \
#    --pretrained_model=../imagenet_resnet50_fusebn/ \
#    --MASK_ON=True > ${log_file} 2>&1 &
#  infer_pid=$!
#  sleep 60
#  kill -9 $infer_pid
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

if [ $index = "mem" ]
then
  echo "Benchmark for $task"
  export FLAGS_fraction_of_gpu_memory_to_use=0.001
  gpu_id=`echo $CUDA_VISIBLE_DEVICES | cut -c1`
  nvidia-smi --id=$gpu_id --query-compute-apps=used_memory --format=csv -lms 100 > gpu_use.log 2>&1 &
  gpu_memory_pid=$!
  $task
  kill $gpu_memory_pid
  awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "Max=", max}' gpu_use.log
else
  echo "Benchmark for $task"
  $task
  if [ ${task} = "train" -a ${num_gpu_devices} -eq 1 ]
  then
      analysis_times 3 39 30
  elif [ ${task} = "train" -a ${num_gpu_devices} -ne 1 ]
  then
      analysis_times 3 37 28
  else
      analysis_times 3 5 5
  fi
fi
