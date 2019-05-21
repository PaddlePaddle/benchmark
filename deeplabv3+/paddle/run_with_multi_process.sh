#!/bin/bash

set -x
#export FLAGS_cudnn_deterministic=true
#export FLAGS_enable_parallel_graph=1

export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fast_eager_deletion_mode=1

DATASET_PATH=${PWD}/data/cityscape/
INIT_WEIGHTS_PATH=${PWD}/deeplabv3plus_xception65_initialize
SAVE_WEIGHTS_PATH=${PWD}/output/model
echo $DATASET_PATH

devices_str=${CUDA_VISIBLE_DEVICES//,/ }
gpu_devices=($devices_str)
num_gpu_devices=${#gpu_devices[*]}

train_crop_size=513
total_step=80
if [ $index = "maxbs" ]; then base_batch_size=9; else base_batch_size=2; fi
batch_size=`expr ${base_batch_size} \* $num_gpu_devices`

python -m paddle.distributed.launch --gpus ${num_gpu_devices} $PWD/train.py \
    --batch_size=${batch_size} \
    --train_crop_size=${train_crop_size} \
    --total_step=${total_step} \
    --init_weights_path=${INIT_WEIGHTS_PATH} \
    --save_weights_path=${SAVE_WEIGHTS_PATH} \
    --dataset_path=${DATASET_PATH} \
    --parallel=True 

