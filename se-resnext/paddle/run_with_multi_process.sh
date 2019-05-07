#!/bin/bash
set -xe
export CUDA_VISIBLE_DEVICES=0,1,2,3

device=${CUDA_VISIBLE_DEVICES//,/ }
arr=($device)
num_gpu_devices=${#arr[*]}

batch_size= 32 #$[32*${num_gpu_devices}]

python -m paddle.distributed.launch --gpus ${num_gpu_devices}  train.py \
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
   --num_epochs=1
