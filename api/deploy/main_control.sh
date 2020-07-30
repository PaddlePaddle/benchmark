#! /bin/bash

function print_usage() {
    echo "Usage:"
    echo "    bash ${0} json_config_dir output_dir gpu_id cpu|gpu|both speed|accuracy|both op_list_file"
    echo ""
    echo "Arguments:"
    echo "  json_config_dir         - the directory of json configs"
    echo "  output_dir              - the output directory"
    echo "  gpu_id (optional)       - the GPU id. Only one GPU can be specified."
    echo "  device (optional)       - cpu, gpu, both"
    echo "  task (optional)         - speed, accuracy, both"
    echo "  op_list_file (optional) - the path which specified op list to test"
}

function print_arguments() {
    echo "Arguments:"
    echo "  $*"
    echo ""
    echo "json_config_dir : ${JSON_CONFIG_DIR}"
    echo "output_dir      : ${OUTPUT_DIR}" 
    echo "gpu_ids         : ${GPU_IDS}"
    echo "device_set      : ${DEVICE_SET[@]}"
    echo "task_set        : ${TASK_SET[@]}"
    echo "op_list_file    : ${OP_LIST_FILE}"
    echo ""
}

export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

OP_BENCHMARK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../" && pwd )"
DEPLOY_DIR="${OP_BENCHMARK_ROOT}/deploy"
TEST_DIR="${OP_BENCHMARK_ROOT}/tests"
export PYTHONPATH=${OP_BENCHMARK_ROOT}:${PYTHONPATH}

if [ $# -lt 2 ]; then
    print_usage
    exit
fi

JSON_CONFIG_DIR=${1}
OUTPUT_DIR=${2}
if [ ! -d ${OUTPUT_DIR} ]; then
    mkdir -p ${OUTPUT_DIR}
fi

GPU_IDS=${3:-"0"}
GPU_IDS_ARRAY=(${GPU_IDS//,/ })
NUM_GPU_DEVICES=${#GPU_IDS_ARRAY[*]}
if [ ${NUM_GPU_DEVICES} -le 0 ]; then
    echo "GPU devices (ids=${GPU_IDS}) should not be empty."
    exit
fi

DEVICE_SET=("gpu" "cpu")
if [ $# -ge 4 ]; then
    if [[ ${4} == "cpu" || ${4} == "gpu" ]]; then
        DEVICE_SET=(${4})
    fi
fi

TASK_SET=("speed" "accuracy")
if [ $# -ge 5 ]; then
    if [[ ${5} == "speed" || ${5} == "accuracy" ]]; then
        TASK_SET=(${5})
    fi
fi

if [ $# -ge 6 ]; then
    OP_LIST_FILE=${6}
else
    OP_LIST_FILE=${OUTPUT_DIR}/api_info.txt
    python ${DEPLOY_DIR}/collect_api_info.py --info_file ${OP_LIST_FILE}
    return_status=$?
    if [ ${return_status} -ne 0 ] || [ ! -f "${OP_LIST_FILE}" ]; then
        OP_LIST_FILE=${DEPLOY_DIR}/api_info.txt
    fi
fi

function print_detail_status() {
    local config_id=$1
    local case_id=$2
    local device=$3
    local backward=$4
    local logfile=$5
    local runtime=$6
    local return_status=$7
    local gpu_id=$8

    if [ ${backward} = "False" ]; then
        local backward_shorten="F"
    else # backward="True"
        local backward_shorten="T"
    fi
    if  [ ${return_status} -eq 0 ]; then
        local run_status="SUCCESS"
    elif [ ${runtime} -ge 600000 ]; then
        local run_status="**TIMEOUT**"
    else
        local run_status="**FAILED**"
    fi
    local print_str="device=${device}, backward=${backward_shorten}, ${logfile}, time=${runtime} ms"
    local print_str_length=${#print_str}
    local timestamp=`date +"%Y-%m-%d %T"`
    if [ ${print_str_length} -lt 80 ]; then
        printf "  [%s][%d-%d][%s] %-80s ...... %s\n" "${gpu_id}" ${config_id} ${case_id} "${timestamp}" "${print_str}" "${run_status}"
    elif [ ${print_str_length} -lt 100 ]; then
        printf "  [%s][%d-%d][%s] %-100s ...... %s\n" "${gpu_id}" ${config_id} ${case_id} "${timestamp}" "${print_str}" "${run_status}"
    else
        printf "  [%s][%d-%d][%s] %-120s ...... %s\n" "${gpu_id}" ${config_id} ${case_id} "${timestamp}" "${print_str}" "${run_status}"
    fi
}

function execute_one_case() {
    local config_id=$1
    local line=$2
    local json_file_path=$3
    local i=$4
    local gpu_id=$5

    local api_name=$(echo $line | cut -d ',' -f1)
    local name=$(echo $line | cut -d ',' -f2)
    local has_backward=$(echo $line | cut -d ',' -f4)
    if [ ${has_backward} = False ]; then  
        direction_set=("forward")
    else
        direction_set=("forward" "backward")
    fi

    local case_id=0
    local case_log="[${config_id}]: api_name=${api_name}, name=${name}, json_file=${json_file_path}, num_configs=${cases_num}, json_id=${i}"
    if [ ${NUM_GPU_DEVICES} -eq 1 ]; then
        echo "${case_log}"
    fi

    # DEVICE_SET is specified by argument: "gpu", "cpu"
    for device in ${DEVICE_SET[@]}; do 
        if [ ${device} = "gpu" ]; then
            local use_gpu="True"
            local repeat=1000
        else
            local use_gpu="False"
            local repeat=100
        fi

        # TASK_SET is specified by argument: "speed", "accuracy"
        for task in "${TASK_SET[@]}"; do 
            if [ ${task} = "accuracy" ]; then
                local framwork_set=("paddle")
            else
                local framwork_set=("paddle" "tensorflow")
            fi
            # framework_set: "paddle", "tensorflow"
            for framework in "${framwork_set[@]}"; do 
                # direction_set: "forward", "backward"
                for direction in "${direction_set[@]}"; 
                do
                    if [ ${direction} = "forward" ]; then
                        local backward="False"
                    else
                        local backward="True"
                    fi

                    case_id=$[$case_id+1]
                    run_cmd="python -m tests.launch ${TEST_DIR}/${name}.py \
                          --api_name ${api_name} \
                          --task ${task} \
                          --framework ${framework} \
                          --json_file ${json_file_path} \
                          --config_id $i \
                          --backward ${backward} \
                          --use_gpu ${use_gpu} \
                          --repeat $repeat \
                          --allow_adaptive_repeat True"

                    run_start=`date +%s%N`
                    if [ "${OUTPUT_DIR}" != "" ]; then
                        logfile=${OUTPUT_DIR}/${api_name}"_"${i}"-"${framework}"_"${device}"_"${task}"_"${direction}".txt"
                        # Set maxmimum runtime to 10min, or it will be considered
                        #  hanged and will be killed.
                        if [ ${device} = "gpu" ]; then
                            CUDA_VISIBLE_DEVICES="${gpu_id}" timeout 600s ${run_cmd} > $logfile 2>&1
                        else
                            CUDA_VISIBLE_DEVICES="" taskset -c ${gpu_id} timeout 600s ${run_cmd} > $logfile 2>&1
                        fi
                        return_status=$?
                    else
                        logfile=""
                        if [ ${device} = "gpu" ]; then
                            CUDA_VISIBLE_DEVICES="${gpu_id}" ${run_cmd}
                        else
                            CUDA_VISIBLE_DEVICES="" taskset -c ${gpu_id} ${run_cmd}
                        fi
                        return_status=$?
                    fi
                    run_end=`date +%s%N`;
                    runtime=$((run_end-run_start))
                    runtime=`expr $runtime / 1000000`

                    if [ ${return_status} -eq 0 ]; then
                        num_success_cases=$[$num_success_cases+1]
                    else
                        num_failed_cases=$[$num_failed_cases+1]
                    fi
                    if [ ${device} == "gpu" ]; then
                        gpu_runtime=`expr $gpu_runtime + $runtime`
                    else
                        cpu_runtime=`expr $cpu_runtime + $runtime`
                    fi
                    local case_log_detail=`print_detail_status ${config_id} ${case_id} "${device}" "${backward}" "${logfile}" ${runtime} ${return_status} ${gpu_id}`
                    if [ ${NUM_GPU_DEVICES} -eq 1 ]; then
                        printf ${case_log_detail}
                    else
                        case_log="${case_log}\n${case_log_detail}"
                    fi
                 done
            done
        done
    done
    if [ ${NUM_GPU_DEVICES} -gt 1 ]; then
        echo -e "${case_log}\n"
    fi
}

function run_all_cases() {
    local gpu_ids_array_index=$1

    local op_info_str=`cat ${OP_LIST_FILE}`
    local op_info_array=(${op_info_str/\\n/ })
    local num_ops=${#op_info_array[*]}

    local num_ops_each_gpu=$((num_ops+NUM_GPU_DEVICES-1))
    local num_ops_each_gpu=$((num_ops_each_gpu/NUM_GPU_DEVICES))
    local config_index_begin=$((gpu_ids_array_index*num_ops_each_gpu))
    local config_index_end=$((config_index_begin+num_ops_each_gpu))
    if [ ${config_index_end} -gt ${num_ops} ]; then
        config_index_end=${num_ops}
    fi

    local config_id=0
    local gpu_id=${GPU_IDS_ARRAY[${gpu_ids_array_index}]}

    echo "config_index_begin: ${config_index_begin}; config_index_end: ${config_index_end}; gpu_id: ${gpu_id}"
    local line_id=${config_index_begin}
    while [ ${line_id} -lt ${config_index_end} ]; do
        local line=${op_info_array[line_id]}
        local json_file=$(echo $line | cut -d ',' -f3)
        if [ "$json_file" != "None" ]; then
            local json_file_path=${JSON_CONFIG_DIR}/${json_file}
            local cases_num=$(grep '"op"' ${json_file_path} | wc -l)
        else
            local cases_num=1
            local json_file_path=None
        fi
    
        for((i=0;i<cases_num;i++)); do
            config_id=$[$config_id+1]
            execute_one_case ${config_id} ${line} ${json_file_path} ${i} ${gpu_id}
        done
        line_id=$((line_id+1))
    done
}

print_arguments $*

cpu_runtime=0
gpu_runtime=0
num_success_cases=0
num_failed_cases=0
for((index=0;index<NUM_GPU_DEVICES;index++)); do
    run_all_cases ${index} &
done
wait

echo "===================================================================="
echo "Summary:"
echo "  ${num_success_cases} successed; ${num_failed_cases} failed"
echo "  GPU runtime: ${gpu_runtime} ms; CPU runtime: ${cpu_runtime} ms"
echo "===================================================================="
