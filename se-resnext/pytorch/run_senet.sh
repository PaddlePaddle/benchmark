#!/bin/bash

set -xe

echo ""
echo "You can use xreki/maskrcnn-benchmark:lastest to test PyTorch 1.1"
echo "Please run nvidia-docker with --shm-size information."
echo "For example:"
echo "  nvidia-docker run --name pytorch_test \\"
echo "          --shm-size 16G --network=host -it --rm \\"
echo "          -v $PWD:/work -w /work \\" 
echo "          xreki/maskrcnn-benchmark:lastest \\"
echo "          bash"
echo "Then you may need install some python packages, use"
echo "          pip install image"
echo "          pip install scipy"
echo ""

#export CUDA_VISIBLE_DEVICES="0"
#export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export WORK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")" && pwd )"
export BENCHMARK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../.." && pwd )"

if [ $# -ne 2 ]; then
  echo "Usage: "
  echo "  CUDA_VISIBLE_DEVICES=0 bash run_senet.sh se_resnext_50|resnet_50 32"
  exit
fi

# 1. task
task=speed

# 2. model: se_resnext_50, resnet_50, resnet_101
model=${1:-se_resnext_50}

# 3. batch_size
base_batchsize=${2:-32}
devices_str=${CUDA_VISIBLE_DEVICES//,/ }
gpu_devices=($devices_str)
num_gpu_devices=${#gpu_devices[*]}
batch_size=`expr $base_batchsize \* $num_gpu_devices`
num_workers=`expr 4 \* $num_gpu_devices`

num_epochs=2

# 4. log_file
log_file=log_senet_${model}_${task}_bs${batch_size}_${num_gpu_devices}


train() {
  echo "Train on ${num_gpu_devices} GPUs"
  echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
  python -c "import torch; print(torch.__version__)"
  gpu_id=`echo $CUDA_VISIBLE_DEVICES |cut -c1`
  nvidia-smi --id=$gpu_id --query-compute-apps=used_memory --format=csv -lms 100 > gpu_use.log 2>&1 &
  gpu_memory_pid=$!
  export PYTHONPATH=${WORK_ROOT}/vision
  stdbuf -oL python ${WORK_ROOT}/train.py \
        --network ${model} \
        --data-dir ImageData/ \
        --batch-size ${batch_size} \
        --num-workers ${num_workers} \
        --num-epochs ${num_epochs} \
        --gpus ${CUDA_VISIBLE_DEVICES} > ${log_file} 2>&1 &
  train_pid=$!
  sleep 600
  kill -9 `ps -ef|grep 'python */SENet/train.py --network'|awk '{print $2}'`
#  kill -9 $train_pid
}

analysis() {
  sed 's/batch\/sec/\ batch\/sec/' ${log_file} > tmp.txt
  python ${BENCHMARK_ROOT}/tools/analysis.py \
    --filename tmp.txt \
    --keyword "Time:" \
    --separator " " \
    --batch_size ${batch_size} \
    --skip_steps 10 \
    --mode 1
  rm tmp.txt
}

train
analysis
awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "Max=", max}' gpu_use.log
