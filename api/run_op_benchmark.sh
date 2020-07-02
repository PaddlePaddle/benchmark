#!/bin/bash

OP_BENCHMARK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")" && pwd )"
DEPLOY_DIR="${OP_BENCHMARK_ROOT}/deploy"
export PYTHONPATH=${OP_BENCHMARK_ROOT}:${PYTHONPATH}

gpu_ids=${1:-"0"}
gpu_ids_array=(${gpu_ids//,/ })
num_gpu_devices=${#gpu_ids_array[*]}

timestamp=`date '+%Y%m%d-%H%M%S'`
output_dir=${OP_BENCHMARK_ROOT}/logs/${timestamp}
if [ ! -d ${OP_BENCHMARK_ROOT}/logs ]; then
    mkdir -p ${OP_BENCHMARK_ROOT}/logs
fi
if [ ! -d ${output_dir} ]; then
    mkdir -p ${output_dir}
fi
 
op_list_file=${output_dir}/api_info.txt
python ${DEPLOY_DIR}/collect_api_info.py --info_file ${op_list_file}
return_status=$?
if [ ${return_status} -ne 0 ] || [ ! -f "${op_list_file}" ]; then
    op_list_file=${DEPLOY_DIR}/api_info.txt
fi
num_ops=`cat ${op_list_file} | wc -l`
num_ops_each_gpu=`expr ${num_ops} / ${num_gpu_devices}`
num_ops_each_gpu=`expr ${num_ops_each_gpu} + 1`

spilt_op_list_prefix="auto_splited_api_info_"
rm -rf ${spilt_op_list_prefix}*
split -l ${num_ops_each_gpu} ${op_list_file} -d -a 1 ${spilt_op_list_prefix}

i=0
config_dir=${OP_BENCHMARK_ROOT}/tests/configs
for gpu_id in ${gpu_ids_array[@]}; do
    echo "Launch on GPU ${gpu_id}"
    log_path=${OP_BENCHMARK_ROOT}/logs/log_${timestamp}_${i}.txt
    bash ${OP_BENCHMARK_ROOT}/deploy/main_control.sh ${config_dir} ${output_dir} ${gpu_id} "both" "both" "${spilt_op_list_prefix}${i}"> ${log_path} 2>&1 &
    i=`expr ${i} + 1`
done
