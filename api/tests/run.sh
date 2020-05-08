#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"
#export GLOG_v=4
#export LD_LIBRARY_PATH=/work/cudnn/cudnn-7.6.5/lib64:${LD_LIBRARY_PATH}

name=${1:-"abs"}
config_id=${2:-"0"}
filename="examples/${name}.json"

python ${name}.py \
      --task "accuracy" \
      --framework "paddle" \
      --json_file ${filename} \
      --config_id ${config_id} \
      --run_with_executor True \
      --check_output False \
      --profiler "none" \
      --backward False \
      --use_gpu False \
      --repeat 1 \
      --log_level 0
