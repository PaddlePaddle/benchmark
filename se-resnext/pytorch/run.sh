#!/bin/bash

set -e

echo ""
echo "You can use pytorch/pytorch:nightly-runtime-cuda9.2-cudnn7 to test PyTorch 1.0"
echo "Please run nvidia-docker with --shm-size information."
echo "For example:"
echo "  nvidia-docker run --name pytorch_test \\"
echo "          --shm-size 16G --network=host -it --rm \\"
echo "          -v $PWD:/work -w /work \\" 
echo "          pytorch/pytorch:nightly-runtime-cuda9.2-cudnn7 \\"
echo "          bash"
echo "Then you may need install some python packages, use"
echo "          pip install image"
echo "          pip install scipy"
echo ""
#pip install --upgrade pytorch torchvision

#export CUDA_VISIBLE_DEVICES="0"
#export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

if [ $# -ne 2 ]; then
  echo "Usage: "
  echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh 32"
  exit
fi

task=speed
base_batchsize=$1
devices_str=${CUDA_VISIBLE_DEVICES//,/ }
gpu_devices=($devices_str)
num_gpu_devices=${#gpu_devices[*]}

batch_size=`expr $base_batchsize \* $num_gpu_devices`
num_epochs=2
num_workers=`expr 2 \* $num_gpu_devices`

log_file=log_${task}_bs${batch_size}_${num_gpu_devices}

if [ ! -d vision ]; then
  git clone https://github.com/pytorch/vision.git
fi
export PYTHONPATH=$PWD/vision

train() {
  echo "Train on ${num_gpu_devices} GPUs"
  echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
  python -c "import torch; print(torch.__version__)"
  gpu_id=`echo $CUDA_VISIBLE_DEVICES |cut -c1`
  nvidia-smi --id=$gpu_id --query-compute-apps=used_memory --format=csv -lms 100 > gpu_use.log 2>&1 &
  gpu_memory_pid=$!
  stdbuf -oL python train.py \
        --network se_resnext_50 \
        --data-dir ImageData/ \
        --batch-size ${batch_size} \
        --num-workers ${num_workers} \
        --num-epochs ${num_epochs} \
        --gpus ${CUDA_VISIBLE_DEVICES} > ${log_file} 2>&1 &
  train_pid=$!
  sleep 300
  kill -9 `ps -ef|grep 'python train.py --network se_resnext_50'|awk '{print $2}'`
#  kill -9 $train_pid
}

analysis_times(){
  sed 's/batch\/sec/\ batch\/sec/' ${log_file} | awk 'BEGIN{count=0}/batch\/sec/{
    step_times[count]=1/$10;
    count+=1;
  }END{
    print "\n================ Benchmark Result ================"
    print "num_epochs:", "'${num_epochs}'"
    print "batch_size:", "'${batch_size}'"
    print "num_workers (used to read data):", "'${num_workers}'"
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
  }'
}

train
analysis_times
awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "Max=", max}' gpu_use.log
