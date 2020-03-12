#!/bin/bash

set -xe

export CUDA_VISIBLE_DEVICES="0"

name=${1:-"abs"}

nvprof python ${name}.py \
      --check_output False \
      --profile False \
      --backward True \
      --use_gpu True \
      --gpu_id 0 \
      --repeat 100 \
      --log_level 1
