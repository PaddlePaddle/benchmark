#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"
#export GLOG_vmodule=operator=4
#export LD_LIBRARY_PATH=/work/cudnn/cudnn-7.6.5/lib64:${LD_LIBRARY_PATH}

NVCC=`which nvcc`
if [ ${NVCC} != "" ]; then
  NVCC_VERSION=`nvcc --version | tail -n 1 | grep "[0-9][0-9]*\.[0-9]" -o | uniq`
  export LD_LIBRARY_PATH=/usr/local/cuda-${NVCC_VERSION}/extras/CUPTI/lib64:${LD_LIBRARY_PATH}
fi

name=${1:-"abs"}
config_id=${2:-"0"}
filename="examples/${name}.json"

python -m launch ${name}.py \
      --task "accuracy" \
      --framework "paddle" \
      --json_file ${filename} \
      --config_id ${config_id} \
      --check_output False \
      --profiler "none" \
      --backward False \
      --use_gpu True \
      --repeat 1 \
      --log_level 0
