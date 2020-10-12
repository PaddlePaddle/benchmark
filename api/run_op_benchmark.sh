#!/bin/bash

OP_BENCHMARK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")" && pwd )"

test_module_name=${1:-"tests"}
gpu_ids=${2:-"0"}

timestamp=`date '+%Y%m%d-%H%M%S'`
if [ ${test_module_name} = "tests" ]; then
    log_dir_name="logs"
elif [ ${test_module_name} = "tests_v2" ]; then
    log_dir_name="logs_v2"
else
    echo "Please set test_module_name to \"tests\" or \"tests_v2\""
    exit
fi
output_dir=${OP_BENCHMARK_ROOT}/${log_dir_name}/${timestamp}
if [ ! -d ${OP_BENCHMARK_ROOT}/${log_dir_name} ]; then
    mkdir -p ${OP_BENCHMARK_ROOT}/${log_dir_name}
fi
if [ ! -d ${output_dir} ]; then
    mkdir -p ${output_dir}
fi

tests_dir=${OP_BENCHMARK_ROOT}/${test_module_name}
config_dir=${OP_BENCHMARK_ROOT}/${test_module_name}/configs
log_path=${OP_BENCHMARK_ROOT}/${log_dir_name}/log_${timestamp}.txt
bash ${OP_BENCHMARK_ROOT}/deploy/main_control.sh ${tests_dir} ${config_dir} ${output_dir} ${gpu_ids} "both" "both" > ${log_path} 2>&1 &
