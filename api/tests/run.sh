#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"
#export GLOG_vmodule=operator=4
#export LD_LIBRARY_PATH=/work/cudnn/cudnn-7.6.5/lib64:${LD_LIBRARY_PATH}

NVCC=`which nvcc`
if [ ${NVCC} != "" ]; then
  NVCC_VERSION=`nvcc --version | tail -n 1 | grep "[0-9][0-9]*\.[0-9]" -o | uniq`
  export LD_LIBRARY_PATH=/usr/local/cuda-${NVCC_VERSION}/extras/CUPTI/lib64:${LD_LIBRARY_PATH}
fi

OP_BENCHMARK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../" && pwd )"
export PYTHONPATH=${OP_BENCHMARK_ROOT}:${PYTHONPATH}

name=${1:-"abs"}
config_id=${2:-"0"}
api_name=${3:-"${name}"}
filename="${OP_BENCHMARK_ROOT}/tests/examples/${name}.json"

python -m tests.launch ${OP_BENCHMARK_ROOT}/tests/${name}.py \
      --task "accuracy" \
      --framework "paddle" \
      --api_name "${api_name}" \
      --json_file ${filename} \
      --config_id ${config_id} \
      --check_output False \
      --profiler "none" \
      --backward True \
      --use_gpu True \
      --repeat 1 \
      --log_level 0
