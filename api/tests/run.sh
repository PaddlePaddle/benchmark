#!/bin/bash

[ -z "$(set | grep '^CUDA_VISIBLE_DEVICES=')" ] && export CUDA_VISIBLE_DEVICES="0"   # Set to "" if testing CPU

NVCC=`which nvcc`
if [ ${NVCC} != "" ]; then
  NVCC_VERSION=`nvcc --version | tail -n 1 | grep "[0-9][0-9]*\.[0-9]" -o | uniq`
  export LD_LIBRARY_PATH=/usr/local/cuda-${NVCC_VERSION}/extras/CUPTI/lib64:${LD_LIBRARY_PATH}
fi

OP_BENCHMARK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../" && pwd )"
export PYTHONPATH=${OP_BENCHMARK_ROOT}:${PYTHONPATH}

name=${1:-"abs"}
config_id=${2:-"0"}
task=${3:-"speed"} # "accuracy" or "speed"

framework="paddle"  # "paddle" or "tensorflow"
filename="${OP_BENCHMARK_ROOT}/tests/configs/${name}.json"
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    use_gpu=False
else
    use_gpu=True
fi

run_args="--task ${task} \
          --framework ${framework} \
          --json_file ${filename} \
          --config_id ${config_id} \
          --check_output False \
          --profiler none \
          --backward True \
          --use_gpu ${use_gpu} \
          --repeat 1 \
          --allow_adaptive_repeat False \
          --log_level 0"

if [ $# -ge 4 ]; then
  api_name=${4}
  run_args="${run_args} \
            --api_name ${api_name}"
fi

python3.6 -m common.launch ${OP_BENCHMARK_ROOT}/tests/${name}.py \
         ${run_args}
