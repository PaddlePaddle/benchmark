#!/bin/bash

set -xe

if [ "$OCR_mode" = "CPU" ]; then
  use_gpu="False"
elif [ "$OCR_mode" = "GPU" ]; then
  use_gpu="True"
elif [ "$OCR_mode" = "MKLDNN" ]; then
  use_gpu="False"
  export FLAGS_use_mkldnn=1
else
  echo "Invalid mode provided. Please use one of {GPU, CPU, MKLDNN}"
  exit 1
fi

source env.sh

python ${OCR_work_root}/paddle/ocr_recognition/infer.py \
    --model ${OCR_model} \
    --model_path ${OCR_model_path} \
    --input_images_list ${OCR_test_list} \
    --input_images_dir ${OCR_test_images} \
    --use_gpu $use_gpu \
    --batch_size ${OCR_batch_size} \
    --iterations ${OCR_iterations} \
    --skip_batch_num ${OCR_skip_batch_num}
