#!/bin/bash

set -xe

export CUDA_VISIBLE_DEVICES="0"

python abs.py \
      --check_output False \
      --profile False \
      --backward False \
      --use_gpu True \
      --gpu_id 0 \
      --repeat 100 \
      --log_level 1
