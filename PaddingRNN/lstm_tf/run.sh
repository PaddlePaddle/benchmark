#!/bin/bash

#nvprof -o timeline_output_medium -f --cpu-profiling off  --profile-from-start off  python  train.py \
export CUDA_VISIBLE_DEVICES=1
python train.py \
    --model_type small
