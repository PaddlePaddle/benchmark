#!bin/bash
set -xe

echo "CUDA_VISIBLE_DEVICES=0 bash run_yolo.sh train|infer 1(speed)|2(mem) /run/log/path"

function _set_params(){
    task="$1"
    index="$2"
    run_log_path=${3:-$(pwd)}
    model_name="yolov3"

    skip_steps=2
    keyword="samples/sec"
    separator=" "
    position=6
    model_mode=1
    run_mode="sp"
    
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    echo $arr
    num_gpu_devices=${#arr[*]}
    #base_batchsize=14 # for max bs
    base_batch_size=8 #origion
    batch_size=`expr ${base_batch_size} \* $num_gpu_devices`
    log_file=${run_log_path}/log_${model_name}_${index}_${num_gpu_devices}
}

function _train(){
    rm *.json
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
    export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
    python3 -u train_yolo3.py \
        --dataset=coco \
        --batch-size=${batch_size} \
        --gpus=${CUDA_VISIBLE_DEVICES} \
        --data-shape=608 \
        --no-random-shape > ${log_file} 2>&1 &
    train_pid=$!
   if [ ${num_gpu_devices} = 1 ]; then
       sleep 600
   elif [ ${num_gpu_devices} = 8 ]; then
       sleep 1000
   else
       sleep 800
   fi

    kill -9 $train_pid
    kill -9 `ps -ef|grep 'train_yolo3'|awk '{print $2}'`
}
