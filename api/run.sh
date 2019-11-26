#!/bin/bash

export CUDA_VISIABLE_DEVICES="0"
#export GLOG_v=4

python abs.py \
      --run_with_executor False \
      --check_output True \
      --profile False \
      --backward False \
      --use_gpu True \
      --repeat 1000 \
      --log_level 0
