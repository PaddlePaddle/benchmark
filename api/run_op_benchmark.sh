#!/bin/bash

OP_BENCHMARK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")" && pwd )"

test_module_name=${1:-"tests"}  # "tests", "tests_v2", "dynamic_tests_v2"
gpu_ids=${2:-"0"}

if [ ${test_module_name} != "tests" ] && [ ${test_module_name} != "tests_v2" ] && [ ${test_module_name} != "dynamic_tests_v2" ]; then
    echo "Please set test_module_name (${test_module_name}) to \"tests\", \"tests_v2\" or \"dynamic_tests_v2\"!"
    exit
fi

install_package() {
    local package_name=$1
    local package_version=$2

    echo "-- Benchmarking module: ${test_module_name}"

    python -c "import ${package_name}" >/dev/null 2>&1
    import_status=$?
    if [ ${import_status} -eq 0 ]; then
        installed_version=`python -c "import ${package_name}; print(${package_name}.__version__)"`
        if [ ${installed_version} == ${package_version} ]; then
            echo "-- ${package_name} ${package_version} is already installed."
        else
            echo "-- Update ${package_name}: ${installed_version} -> ${package_version}"
            pip install -U ${package_name}==${package_version}
        fi
    else
        echo "-- Install ${package_name} ${package_version}"
        pip install ${package_name}==${package_version}
    fi
}

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
    install_package "torch" "1.9.0"
else
    testing_mode="static"
    install_package "tensorflow" "2.3.1"
fi

tests_dir=${OP_BENCHMARK_ROOT}/${test_module_name}
log_path=${OUTPUT_ROOT}/log_${test_module_name}_${timestamp}.txt
bash ${OP_BENCHMARK_ROOT}/deploy/main_control.sh ${tests_dir} ${config_dir} ${output_dir} ${gpu_ids} "both" "both" "none" "both" "${testing_mode}" > ${log_path} 2>&1 &
