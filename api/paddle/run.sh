#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"
#export GLOG_v=4

name=${1:-"abs"}

nvprof python ${name}.py \
      --run_with_executor True \
      --check_output False \
      --profiler "none" \
      --backward True \
      --use_gpu True \
      --repeat 100 \
      --log_level 1
