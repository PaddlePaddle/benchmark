#!/usr/bin/env bash
WEIGHTS_PATH=${PWD}/deeplabv3plus_gn

python ./eval.py \
    --init_weights=$WEIGHTS_PATH \
    --norm_type=gn \
    --dataset_path=$DATASET_PATH
