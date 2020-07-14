#!/bin/bash

OP_BENCHMARK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")" && pwd )"

gpu_ids=${1:-"0"}

timestamp=`date '+%Y%m%d-%H%M%S'`
output_dir=${OP_BENCHMARK_ROOT}/logs/${timestamp}
if [ ! -d ${OP_BENCHMARK_ROOT}/logs ]; then
    mkdir -p ${OP_BENCHMARK_ROOT}/logs
fi
if [ ! -d ${output_dir} ]; then
    mkdir -p ${output_dir}
fi

config_dir=${OP_BENCHMARK_ROOT}/tests/configs
log_path=${OP_BENCHMARK_ROOT}/logs/log_${timestamp}.txt
bash ${OP_BENCHMARK_ROOT}/deploy/main_control.sh ${config_dir} ${output_dir} ${gpu_ids} "both" "both" > ${log_path} 2>&1 &
