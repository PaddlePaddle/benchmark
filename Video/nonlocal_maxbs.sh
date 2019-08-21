#!/usr/bin/env bash

# settings
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
model_path="/Paddle/Models"

# args
min=$1
max=$2

# flags
export FLAGS_eager_delete_tensor_gb=0
export FLAGS_fraction_of_gpu_memory_to_use=1 

# paths
log_file_prefix='nextvlad-bs'
video_path="${model_path}/PaddleCV/PaddleVideo"

train(){
    batch_size=$1
    log_file="${log_file_prefix}${batch_size}.log"
    dir=$(pwd)
    cd ${video_path}
    train_cmd=" --model_name=NONLOCAL \
	--config=${video_path}/configs/nonlocal.txt \
	--epoch=1 \
	--valid_interval=1 \
	--log_interval=10"
    train_cmd="python train.py "${train_cmd}
    ${train_cmd} > ${dir}/${log_file} 2>&1 &
    pid=$!
    wait $pid
    cd ${dir}
    error_string="Please shrink FLAGS_fraction_of_gpu_memory_to_use or FLAGS_initial_gpu_memory_in_mb or FLAGS_reallocate_gpu_memory_in_mbenvironment variable to a lower value"
    if [ `grep -c "${error_string}" ${log_file}` -eq 0 ]; then
      return 0
    else
      return 1
    fi
}

while [ $min -lt $max ]; do
    current=`expr '(' "${min}" + "${max}" + 1 ')' / 2`
    echo "Try batchsize=${current}"
    if train ${current}; then
        min=${current}
    else
        max=`expr ${current} - 1`
    fi
done

echo "Maximum batchsize is ${min}"
