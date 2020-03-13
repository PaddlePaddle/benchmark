#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"
#export GLOG_v=4
#export LD_LIBRARY_PATH=/work/cudnn-7.6/lib64:${LD_LIBRARY_PATH}

name=${1:-"abs"}

python ${name}.py \
      --dtype "float32" \
      --run_with_executor True \
      --check_output False \
      --profiler "none" \
      --backward False \
      --use_gpu True \
      --repeat 100 \
      --log_level 0
