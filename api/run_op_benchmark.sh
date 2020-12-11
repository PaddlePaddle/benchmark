#!/bin/bash

OP_BENCHMARK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")" && pwd )"

test_module_name=${1:-"tests"}  # "tests", "tests_v2", "dynamic_tests_v2"
gpu_ids=${2:-"0"}

if [ ${test_module_name} != "tests" ] && [ ${test_module_name} != "tests_v2" ] && [ ${test_module_name} != "dynamic_tests_v2" ]; then
    echo "Please set test_module_name to \"tests\", \"tests_v2\" or \"dynamic_tests_v2\"!"
    exit
fi

OUTPUT_ROOT=${OP_BENCHMARK_ROOT}/logs
if [ ! -d ${OUTPUT_ROOT} ]; then
    mkdir -p ${OUTPUT_ROOT}
fi

timestamp=`date '+%Y%m%d-%H%M%S'`
output_dir=${OUTPUT_ROOT}/${test_module_name}/${timestamp}
if [ ! -d ${output_dir} ]; then
    mkdir -p ${output_dir}
fi

if [ ${test_module_name} = "tests" ]; then
    config_dir=${OP_BENCHMARK_ROOT}/tests/configs
else
    config_dir=${OP_BENCHMARK_ROOT}/tests_v2/configs
fi

if [ "${test_module_name}" == "dynamic_tests_v2" ]; then
    testing_mode="dynamic"
else
    testing_mode="static"
fi

tests_dir=${OP_BENCHMARK_ROOT}/${test_module_name}
log_path=${OUTPUT_ROOT}/log_${test_module_name}_${timestamp}.txt
bash ${OP_BENCHMARK_ROOT}/deploy/main_control.sh ${tests_dir} ${config_dir} ${output_dir} ${gpu_ids} "both" "both" "none" "both" "${testing_mode}" > ${log_path} 2>&1 &
