#!/bin/bash

set -x
echo " CUDA_VISIBLE_DEVICES=7 bash run_deeplabv3.sh speed sp /log/file"
#export CUDA_VISIBLE_DEVICES="0,1,2,3"
#export CUDA_VISIBLE_DEVICES="0"

function _set_params(){
    export TF_MODELS_ROOT=/ssd3/heya/tensorflow/tensorflow_models/models/research # NOTE: this is in 19 local
    
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}
    
    train_crop_size=513
    total_step=80
    base_batch_size=2
    batch_size=`expr ${base_batch_size} \* ${num_gpu_devices}`
    
    index=${1}
    run_mode=${2}
    run_log_path=${3:-$(pwd)}
    model_name=deeplabv3
    
    log_file=${run_log_path}/log_${model_name}_${index}_${num_gpu_devices}
    skip_steps=3
    keyword="learning.py:507"
    separator=" "
    position=10
    range=2:6
    model_mode=0
    run_mode=sp
}

function _train(){
  echo "Train on ${num_gpu_devices} GPUs"
  export PYTHONPATH=${TF_MODELS_ROOT}:${TF_MODELS_ROOT}/slim
  if [ ${index} = "mem" ]; then
      sed -i '369c \    session_config.gpu_options.allow_growth = True' ${TF_MODELS_ROOT}/deeplab/train.py
  elif [ ${index} = "speed" ]; then
      sed -i '369c \    session_config.gpu_options.allow_growth = False' ${TF_MODELS_ROOT}/deeplab/train.py
  fi
  rm -rf ./train_logs

  python ${TF_MODELS_ROOT}/deeplab/train.py  \
           --logtostderr  \
           --training_number_of_steps=${total_step}  \
           --train_split=train  \
           --model_variant=xception_65  \
           --atrous_rates=6  \
           --atrous_rates=12  \
           --atrous_rates=18  \
           --output_stride=16  \
           --decoder_output_stride=4  \
           --train_crop_size=${train_crop_size}  \
           --train_crop_size=${train_crop_size}  \
           --train_batch_size=${batch_size}  \
           --dataset=cityscapes  \
           --train_logdir=./train_logs  \
           --dataset_dir=/ssd1/ljh/dataset/tf_cityscapes/tfrecord/  \
           --tf_initial_checkpoint=${TF_MODELS_ROOT}/deeplab/deeplabv3_cityscapes_train/model.ckpt.index  \
           --num_clones ${num_gpu_devices} > ${log_file} 2>&1 &
  train_pid=$!
  sleep 400
  kill =9 $train_pid
}

source ${BENCHMARK_ROOT}/competitive_products/common_scripts/run_model.sh
_set_params $@
_run
