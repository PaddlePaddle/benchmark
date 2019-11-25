#!/bin/bash

export CUDA_VISIABLE_DEVICES="0"

python abs.py \
      --backward False \
      --use_gpu True \
      --repeat 1000 \
      --log_level 0
