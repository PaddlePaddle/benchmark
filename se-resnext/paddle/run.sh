#!/bin/bash
if [ $# -ne 2 ]; then
  echo "Usage: "
  echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh speed 16"
  exit
fi

base_batchsize=$2

fun(){
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    gpus=${#arr[*]}
    batch_size=`expr $base_batchsize \* $gpus`
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$gpus, batch_size=$batch_size"

    python train.py \
       --model=SE_ResNeXt50_32x4d \
       --batch_size=$batch_size \
       --total_images=1281167 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --model_save_dir=output/ \
       --with_mem_opt=False \
       --lr_strategy=piecewise_decay \
       --lr=0.1 > log 2>&1 &
    train_pid=$!
    sleep 300
    kill -9 $train_pid
}
if [ $1 = 'mem' ]
then
    echo "test for $1"
    export FLAGS_fraction_of_gpu_memory_to_use=0.001
    gpu_id=`echo $CUDA_VISIBLE_DEVICES |cut -c1`
    nvidia-smi --id=$gpu_id --query-compute-apps=used_memory --format=csv -lms 100 > gpu_use.log 2>&1 &
    gpu_memory_pid=$!
    fun
    kill $gpu_memory_pid
    awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "Max=", max}' gpu_use.log
else
    echo "test for $1"
    fun
    awk 'BEGIN{count=0} {if(NF==14){count+=1;{if(count>5){print $0;res_c+=1;res_time+=$13;}}}}END{print "all_step:",res_c,"\tavg_time:",(res_time/res_c)}' log
fi