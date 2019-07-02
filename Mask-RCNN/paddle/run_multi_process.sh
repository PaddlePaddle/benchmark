#!bin/bash
set -xe

#export FLAGS_cudnn_deterministic=true
#export FLAGS_enable_parallel_graph=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98
export FLAGS_memory_fraction_of_eager_deletion=1.0
export FLAGS_conv_workspace_size_limit=1500

base_batch_size=1

export CUDA_VISIBLE_DEVICES=0,1

device=${CUDA_VISIBLE_DEVICES//,/ }
arr=($device)
num_gpu_devices=${#arr[*]}

python -m paddle.distributed.launch --log_dir=./mylog --selected_gpus=$CUDA_VISIBLE_DEVICES train.py \
            --model_save_dir=output/ \
            --pretrained_model=../imagenet_resnet50_fusebn/ \
            --data_dir=./dataset/coco \
            --im_per_batch=${base_batch_size} \
            --MASK_ON=True