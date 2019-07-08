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

export CUDA_VISIBLE_DEVICES="0"
export WORK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")" && pwd )"
export BENCHMARK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../.." && pwd )"

# 1. task
task=speed

# 2. batch_size
base_batchsize=32
devices_str=${CUDA_VISIBLE_DEVICES//,/ }
gpu_devices=($devices_str)
num_gpu_devices=${#gpu_devices[*]}
batch_size=`expr $base_batchsize \* $num_gpu_devices`
num_workers=`expr 4 \* $num_gpu_devices`

# 3. model: resnet50, resnet101
model=resnet50

# 4. data_path
data_path=/data/ILSVRC2012/

# 5. log_file
log_file=log_examples_${model}_${task}_bs${batch_size}_${num_gpu_devices}

train() {
  echo "Train on ${num_gpu_devices} GPUs"
  echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
  python -c "import torch; print(torch.__version__)"

  export PYTHONPATH=$WORK_ROOT/vision
  python ${WORK_ROOT}/examples/imagenet/main.py \
         --arch ${model} \
         --workers ${num_workers} \
         --epochs 1 \
         --batch-size ${batch_size} \
         --resume ./output/examples \
         --print-freq 10 \
         --gpu ${CUDA_VISIBLE_DEVICES} \
         ${data_path} > ${log_file} 2>&1 &
  train_pid=$!
  sleep 600
  kill -9 $train_pid
}

analysis() {
  python ${BENCHMARK_ROOT}/tools/analysis.py \
    --filename ${log_file} \
    --keyword "Time" \
    --separator " " \
    --batch_size ${batch_size} \
    --skip_steps 10 \
    --mode 0
}

train
analysis
