#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"
#export GLOG_v=4

python abs.py \
      --run_with_executor True \
      --check_output False \
      --profiler "none" \
      --backward False \
      --use_gpu True \
      --repeat 100 \
      --log_level 1
