#!/bin/bash

set -xe

core_num=`lscpu |grep -m1 "CPU(s)"|awk -F':' '{print $2}'|xargs`
if [ "$OCR_mode" = "CPU" ]; then
  if [ $core_num -gt $batch_size ]; then
    echo "Batch size should be greater or equal to the number of 
          available cores, when parallel mode is set to True."
  fi
  use_gpu="False"
  save_model_dir="${OCR_work_root}/paddle/results/cpu_${OCR_model}_models_${OCR_batch_size}"
elif [ "$OCR_mode" = "GPU" ]; then
  use_gpu="True"
  save_model_dir="${OCR_work_root}/paddle/results/gpu_${OCR_model}_models_${OCR_batch_size}"
elif [ "$OCR_mode" = "MKLDNN" ]; then
  use_gpu="False"
  save_model_dir="${OCR_work_root}/paddle/results/mkldnn_${OCR_model}_models_${OCR_batch_size}"
  parallel="False"
  export FLAGS_use_mkldnn=1
else
  echo "Invalid mode provided. Please use one of {GPU, CPU, MKLDNN}"
  exit 1
fi

source env.sh

python ${OCR_work_root}/paddle/ocr_recognition/train.py \
    --model ${OCR_model} \
    --init_model ${OCR_init_model} \
    --train_images ${OCR_train_images} \
    --train_list ${OCR_train_list} \
    --test_images ${OCR_test_images} \
    --test_list ${OCR_test_list} \
    --save_model_dir ${save_model_dir} \
    --use_gpu ${use_gpu} \
    --parallel ${OCR_parallel} \
    --batch_size ${OCR_batch_size} \
    --log_period ${OCR_log_period} \
    --save_model_period ${OCR_save_model_period} \
    --eval_period ${OCR_eval_period} \
    --total_step ${OCR_total_step}

