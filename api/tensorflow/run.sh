#!/bin/bash

set -xe

export CUDA_VISIBLE_DEVICES="1"

nvprof python abs.py \
      --check_output False \
      --profile False \
      --backward False \
      --use_gpu True \
      --gpu_id 0 \
      --repeat 10 \
      --log_level 1
