#!/bin/bash

OP_BENCHMARK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")" && pwd )"
DEPLOY_DIR="${OP_BENCHMARK_ROOT}/deploy"
export PYTHONPATH=${OP_BENCHMARK_ROOT}:${PYTHONPATH}

gpu_ids=${1:-"0"}

timestamp=`date '+%Y%m%d-%H%M%S'`
#output_dir=${OP_BENCHMARK_ROOT}/logs/${timestamp}
output_dir="test"
if [ ! -d ${OP_BENCHMARK_ROOT}/logs ]; then
    mkdir -p ${OP_BENCHMARK_ROOT}/logs
fi
if [ ! -d ${output_dir} ]; then
    mkdir -p ${output_dir}
fi

test_op_list="/work/benchmark/api/op_list.txt"
config_dir=${OP_BENCHMARK_ROOT}/tests/configs
log_path=${OP_BENCHMARK_ROOT}/logs/log_${timestamp}_${i}.txt
bash ${OP_BENCHMARK_ROOT}/deploy/main_control.sh ${config_dir} ${output_dir} ${gpu_ids} "both" "both" "${test_op_list}" #> ${log_path} 2>&1 &
