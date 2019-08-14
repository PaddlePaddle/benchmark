#!/bin/bash

set -xe

gpu_id="-1"
if [ $# -le 1 ]; then
  gpu_id="$1"
fi

if [ ${gpu_id} -eq "-1" ]; then
  USE_GPU=false
  export CUDA_VISIBLE_DEVICES=""
else
  USE_GPU=true
  export CUDA_VISIBLE_DEVICES="${gpu_id}"
fi

MODEL_DIR=/work/inference/ernie/model
DATA_FILE=/work/inference/ernie/seq128_data/test_ds
REPEAT=1

./build/inference --logtostderr \
    --model_dir ${MODEL_DIR} \
    --data ${DATA_FILE} \
    --repeat ${REPEAT} \
    --use_gpu ${USE_GPU} \
    --use_analysis true \
    --print_outputs true
