#!bin/bash
set -xe

#export FLAGS_cudnn_deterministic=true
#export FLAGS_enable_parallel_graph=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98
export FLAGS_memory_fraction_of_eager_deletion=1.0

device=${CUDA_VISIBLE_DEVICES//,/ }
arr=($device)
num_gpu_devices=${#arr[*]}
if [ $index = "maxbs" ]; then base_batch_size=14; else base_batch_size=8; fi
batch_size=`expr ${base_batch_size} \* $num_gpu_devices`
log_file=${run_log_path}/${model_name}_${task}_${index}_${num_gpu_devices}

python -m paddle.distributed.launch --gpus ${num_gpu_devices} train.py \
 --model_save_dir=output/ \
 --pretrain=./weights/darknet53/ \
 --data_dir=./dataset/coco/ \
 --batch_size=${base_batch_size} 

