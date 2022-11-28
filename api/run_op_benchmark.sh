#!/bin/bash

OP_BENCHMARK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")" && pwd )"

test_module_name=${1:-"tests"}  # "tests", "tests_v2"
gpu_ids=${2:-"0"}
op_type=${3:-"all"}  # "all" or specified op_type, such as elementwise
device_type=${4:-"both"}
precision=${5:-"fp32"}
task=${6:-"both"}
framework=${7:-"both"}

if [ ${test_module_name} != "tests" ] && [ ${test_module_name} != "tests_v2" ]; then
  echo "Please set test_module_name (${test_module_name}) to \"tests\", \"tests_v2\"!"
  exit
fi

if [ ${device_type} != "both" ] && [ ${device_type} != "gpu" ] && [ ${device_type} != "cpu" ]; then
  echo "Please set device_type (${device_type}) to \"both\", \"cpu\" or \"gpu\"!"
  exit
fi

if [ ${precision} != "both" ] && [ ${precision} != "fp32" ] && [ ${precision} != "fp16" ]; then
  echo "Please set precision (${precision}) to \"both\", \"fp32\" or \"fp16\"!"
  exit
fi

#export FLAGS_use_autotune=1
#export GLOG_vmodule=switch_autotune=3

install_package() {
  local package_name=$1
  local package_version=$2

  echo "-- Benchmarking module: ${test_module_name}"

  python -c "import ${package_name}" >/dev/null 2>&1
  import_status=$?
  if [ ${import_status} -eq 0 ]; then
    installed_version=`python -c "import ${package_name}; print(${package_name}.__version__)"`
    if [[ "${installed_version}" > "${package_version}" ]]; then
      echo "-- ${package_name} ${installed_version} (newer than ${package_version}) is already installed."
    elif [ "${installed_version}" == "${package_version}" ]; then
      echo "-- ${package_name} ${package_version} is already installed."
    else
      if [ "${package_version}" != "" ]; then
        echo "-- Update ${package_name}: ${installed_version} -> ${package_version}"
        pip install -U ${package_name}==${package_version}
      else
        pip install -U ${package_name}
      fi
    fi
  else
    echo "-- Install ${package_name} ${package_version}"
    pip install ${package_name}==${package_version}
  fi
}

run_op_benchmark() {
  local testing_mode=$1

  OUTPUT_ROOT=${OP_BENCHMARK_ROOT}/logs
  if [ ! -d ${OUTPUT_ROOT} ]; then
    mkdir -p ${OUTPUT_ROOT}
  fi
  
  timestamp=`date '+%Y%m%d-%H%M%S'`
  if [ ${test_module_name} == "tests" ]; then
    output_dir=${OUTPUT_ROOT}/${test_module_name}_${testing_mode}/${timestamp}
  else
    output_dir=${OUTPUT_ROOT}/${test_module_name}/${timestamp}
  fi
  if [ ! -d ${output_dir} ]; then
    mkdir -p ${output_dir}
  fi
  echo "-- output_dir: ${output_dir}"
  
  config_dir=${OP_BENCHMARK_ROOT}/tests_v2/configs
  echo "-- config_dir: ${config_dir}"
  
  tests_dir=${OP_BENCHMARK_ROOT}/${test_module_name}
  echo "-- tests_dir: ${tests_dir}"
  log_path=${OUTPUT_ROOT}/log_${test_module_name}_${timestamp}.txt
  bash ${OP_BENCHMARK_ROOT}/deploy/main_control.sh ${tests_dir} ${config_dir} ${output_dir} ${gpu_ids} ${device_type} "both" "none" "both" "${testing_mode}" "None" "${precision}" > ${log_path} 2>&1 &
}

run_specified_op() {
  local testing_mode=$1

  OUTPUT_ROOT=${OP_BENCHMARK_ROOT}/logs/${op_type}
  if [ ! -d ${OUTPUT_ROOT} ]; then
    mkdir -p ${OUTPUT_ROOT}
  fi

  timestamp=`date '+%Y%m%d-%H%M%S'`
  if [ ${test_module_name} == "tests" ]; then
    output_dir=${OUTPUT_ROOT}/${test_module_name}_${testing_mode}/${timestamp}
  else
    output_dir=${OUTPUT_ROOT}/${test_module_name}/${timestamp}
  fi
  if [ ! -d ${output_dir} ]; then
    mkdir -p ${output_dir}
  fi
  echo "-- output_dir: ${output_dir}"
  
  config_dir=${OP_BENCHMARK_ROOT}/tests_v2/op_configs
  echo "-- config_dir: ${config_dir}"
 
  tests_dir=${OP_BENCHMARK_ROOT}/${test_module_name}
  echo "-- tests_dir: ${tests_dir}"
  log_path=${OUTPUT_ROOT}/log_${test_module_name}_${timestamp}.txt
  bash ${OP_BENCHMARK_ROOT}/deploy/main_control.sh ${tests_dir} ${config_dir} ${output_dir} "${gpu_ids}" "gpu" "${task}" "none" "${framework}" "${testing_mode}" "${op_type}" "${precision}" > ${log_path} 2>&1 &
}

main() {
  if [ ${test_module_name} == "tests" ]; then
    testing_mode="dynamic"
    # For ampere, need to install the nightly build cuda11.3 version using the following command:
    # pip install --pre torch -f https://download.pytorch.org/whl/nightly/cu113/torch_nightly.html
    install_package "torch" "1.12.0"
    install_package "torchvision"
  else
    testing_mode="static"
    install_package "tensorflow" "2.3.1"
  fi

  case ${op_type} in
    all)
      run_op_benchmark ${testing_mode}
      ;;
    *)
      run_specified_op ${testing_mode}
      ;;
  esac
}

main
