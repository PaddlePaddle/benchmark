#! /bin/bash

function print_usage() {
    echo "Usage:"
    echo "    bash ${0} test_dir json_config_dir output_dir gpu_id cpu|gpu|both speed|accuracy|both op_list_file framework testing_mode precision"
    echo ""
    echo "Arguments:"
    echo "   1. test_dir                - the directory of tests case"
    echo "   2. json_config_dir         - the directory of json configs"
    echo "   3. output_dir              - the output directory"
    echo "   4. gpu_id (optional)       - the GPU id. Only one GPU can be specified."
    echo "   5. device (optional)       - cpu, gpu, both"
    echo "   6. task (optional)         - speed, accuracy, both"
    echo "   7. op_list_file (optional) - the path which specified op list to test"
    echo "   8. framework (optional)    - paddle, tensorflow, pytorch, both"
    echo "   9. testing_mode (optional) - the testing_mode of paddle, dynamic(default)|static."
    echo "  10. op_name (optional)      - specified op name or list string. such as conv2d | conv1d,conv2d,conv3d."
    echo "  11. precision (optional)    - the precision to test, fp32|fp16|both"
}

function print_arguments() {
    echo "Arguments:"
    echo "  $*"
    echo ""
    echo "test_dir        : ${TEST_DIR}"
    echo "json_config_dir : ${JSON_CONFIG_DIR}"
    echo "output_dir      : ${OUTPUT_DIR}" 
    echo "gpu_ids         : ${GPU_IDS}"
    echo "device_set      : ${DEVICE_SET[@]}"
    echo "task_set        : ${TASK_SET[@]}"
    echo "op_list_file    : ${OP_LIST_FILE}"
    echo "framework       : ${FRAMEWORK_SET[@]}"
    echo "testing_mode    : ${TESTING_MODE}"
    echo "op_name         : ${OP_NAME}"
    echo "precision       : ${PRECISION_SET}"
    echo ""
}

declare -A DEVICE_TASK_PID_MAP
declare -A TASK_PID_RUN_START_MAP
declare -A TASK_PID_INFO_MAP
declare -A TASK_PID_DETAIL_CONTENT_MAP
declare -A TASK_PID_DETAIL_KEY_MAP
declare -A DETAIL_KEY_TASK_PIDS_MAP
declare -A DETAIL_KEY_TOTAL_TASK_NUM_MAP
declare -A DETAIL_KEY_FINISHED_TASK_NUM_MAP
declare -A DETAIL_KEY_CONTENT_MAP

export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

OP_BENCHMARK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../" && pwd )"
DEPLOY_DIR="${OP_BENCHMARK_ROOT}/deploy"
export PYTHONPATH=${OP_BENCHMARK_ROOT}:${PYTHONPATH}

if [ $# -lt 3 ]; then
    print_usage
    exit
fi

TEST_DIR=${1}
TEST_MODULE_NAME=${TEST_DIR##*/}

JSON_CONFIG_DIR=${2}
OUTPUT_DIR=${3}
if [ ! -d ${OUTPUT_DIR} ]; then
    mkdir -p ${OUTPUT_DIR}
fi

GPU_IDS=${4:-"0"}
GPU_IDS_ARRAY=(${GPU_IDS//,/ })
NUM_GPU_DEVICES=${#GPU_IDS_ARRAY[*]}
for i in ${GPU_IDS_ARRAY[*]}; do
    DEVICE_TASK_PID_MAP[$i]=0
done
if [ ${NUM_GPU_DEVICES} -le 0 ]; then
    echo "GPU devices (ids=${GPU_IDS}) should not be empty."
    exit
fi

DEVICE_SET=("gpu" "cpu")
if [ $# -ge 5 ]; then
    if [[ ${5} == "cpu" || ${5} == "gpu" ]]; then
        DEVICE_SET=(${5})
    fi
fi

TASK_SET=("speed" "accuracy")
if [ $# -ge 6 ]; then
    if [[ ${6} == "speed" || ${6} == "accuracy" ]]; then
        TASK_SET=(${6})
    fi
fi

if [ $# -ge 7 ]; then
    OP_LIST_FILE=${7}
    if [ ! -f "${OP_LIST_FILE}" ]; then
        echo "The specified OP_LIST_FILE (${OP_LIST_FILE}) does not exist in the filesystem!"
        unset OP_LIST_FILE
    fi
fi
if [ "${OP_LIST_FILE}" == "" ]; then
    OP_NAME="None"
    if [ $# -ge 10 ]; then
        OP_NAME=${10}
    fi
    OP_LIST_FILE=${OUTPUT_DIR}/api_info.txt
    python ${DEPLOY_DIR}/collect_api_info.py \
        --test_module_name ${TEST_MODULE_NAME} \
        --specified_op_list ${OP_NAME} \
        --info_file ${OP_LIST_FILE}
    return_status=$?
    if [ ${return_status} -ne 0 ] || [ ! -f "${OP_LIST_FILE}" ]; then
        OP_LIST_FILE=${DEPLOY_DIR}/api_info.txt
    fi
fi

TESTING_MODE="dynamic"
FRAMEWORK_SET=("paddle" "pytorch")
if [ $# -ge 8 ]; then
    if [ $# -ge 9 ]; then
        TESTING_MODE=${9}
    elif [ ${TEST_MODULE_NAME} == "dynamic_tests_v2" ]; then
        TESTING_MODE="dynamic"
    elif [ ${TEST_MODULE_NAME} == "tests_v2" ]; then
        TESTING_MODE="static"
    fi

    if [ ${TESTING_MODE} == "static" ]; then
        if [[ ${8} == "paddle"  || ${8} == "tensorflow" ]]; then
            FRAMEWORK_SET=(${8})
        elif [ ${8} == "both" ]; then
            FRAMEWORK_SET=("paddle" "tensorflow")
        else
            echo "The static testing mode only can test paddle or tensorflow."
        fi
    elif [ ${TESTING_MODE} == "dynamic" ]; then
        if [[ ${8} == "paddle"  || ${8} == "pytorch" ]]; then
            FRAMEWORK_SET=(${8})
        elif [ ${8} == "both" ]; then
            FRAMEWORK_SET=("paddle" "pytorch")
        else
            echo "The dynamic testing mode only can test paddle or pytorch."
        fi
    fi
fi

PRECISION_SET=("fp32")
if [ $# -ge 11 ]; then
    if [[ ${11} == "fp32" || ${11} == "fp16" ]]; then
        PRECISION_SET=(${11})
    elif [[ ${11} == "both" ]]; then
        PRECISION_SET=("fp32" "fp16")
    fi
fi

function print_detail_status() {
    local config_id=$1
    local case_id=$2
    local device=$3
    local backward=$4
    local logfile=$5
    local gpu_id=$6
    local runtime=$7
    local return_status=$8

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
    [ "$BENCHMARK_PRINT_FAIL_LOG" == "1" ] && cat ${logfile}
}

function print_finished_task_detail() {
    local finished_task_pid=$1
    local finished_run_end=$2
    local detail_key=${TASK_PID_DETAIL_KEY_MAP[$finished_task_pid]}
    [ -z "${TASK_PID_INFO_MAP[$finished_task_pid]}" ] && return 0
    finished_run_start=TASK_PID_RUN_START_MAP[$finished_task_pid]
    runtime=$((finished_run_end-finished_run_start))
    runtime=`expr $runtime / 1000000`
    wait $finished_task_pid
    return_status=$?
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
    TASK_PID_DETAIL_CONTENT_MAP[$finished_task_pid]=$(print_detail_status ${TASK_PID_INFO_MAP[$finished_task_pid]} ${runtime} ${return_status})
    DETAIL_KEY_FINISHED_TASK_NUM_MAP[$detail_key]=$[${DETAIL_KEY_FINISHED_TASK_NUM_MAP[$detail_key]}+1]
    if [ ${DETAIL_KEY_FINISHED_TASK_NUM_MAP[$detail_key]} -eq ${DETAIL_KEY_TOTAL_TASK_NUM_MAP[$detail_key]} ]
    then
        detail_content="${DETAIL_KEY_CONTENT_MAP[$detail_key]}"
        for task_pid in ${DETAIL_KEY_TASK_PIDS_MAP[$detail_key]}
        do
            detail_content="${detail_content}\n${TASK_PID_DETAIL_CONTENT_MAP[$task_pid]}"
            unset TASK_PID_DETAIL_CONTENT_MAP[$task_pid]
        done
        echo -e "${detail_content}\n"
        unset DETAIL_KEY_CONTENT_MAP[$detail_key]
        unset DETAIL_KEY_TASK_PIDS_MAP[$detail_key]
        unset DETAIL_KEY_TOTAL_TASK_NUM_MAP[$detail_key]
        unset DETAIL_KEY_FINISHED_TASK_NUM_MAP[$detail_key]
    fi
    unset TASK_PID_INFO_MAP[$finished_task_pid]
    unset TASK_PID_RUN_START_MAP[$finished_task_pid]
    unset TASK_PID_DETAIL_KEY_MAP[$finished_task_pid]
}

function execute_one_case() {
    local config_id=$1
    local line=$2
    local json_file_path=$3
    local i=$4

    local api_name=$(echo $line | cut -d ',' -f1)
    local name=$(echo $line | cut -d ',' -f2)
    local has_backward=$(echo $line | cut -d ',' -f4)
    if [ ${has_backward} = False ]; then  
        direction_set=("forward")
    else
        direction_set=("forward" "backward")
    fi

    local case_id=0
    local detail_key="${config_id}-${i}"
    local task_pids=""
    DETAIL_KEY_TOTAL_TASK_NUM_MAP[$detail_key]=-1
    DETAIL_KEY_FINISHED_TASK_NUM_MAP[$detail_key]=0
    DETAIL_KEY_CONTENT_MAP[$detail_key]="[${config_id}]: api_name=${api_name}, name=${name}, json_file=${json_file_path}, num_configs=${cases_num}, json_id=${i}"

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
            # FRAMEWORK_SET is specified by argument: "paddle", "tensorflow"
            for framework in "${FRAMEWORK_SET[@]}"; do 
                [ "${task}" == "accuracy" -a "${framework}" == "tensorflow" ] && continue
                [ "${task}" == "accuracy" -a "${framework}" == "pytorch" ] && continue
                # direction_set: "forward", "backward"
                for direction in "${direction_set[@]}"; 
                do
                    if [ ${direction} = "forward" ]; then
                        local backward="False"
                    else
                        local backward="True"
                    fi

                    for precision in "${PRECISION_SET[@]}"; 
                    do
                        [ "${device}" == "cpu" -a "${precision}" == "fp16" ] && continue

                        case_id=$[$case_id+1]
                        if [ "${TEST_MODULE_NAME}" = "tests" ]; then
                            test_script="${TEST_DIR}/test_main.py --filename ${name}"
                        else
                            test_script="${TEST_DIR}/${name}.py"
                        fi
                        run_cmd="python -m common.launch ${test_script} \
                              --api_name ${api_name} \
                              --task ${task} \
                              --framework ${framework} \
                              --testing_mode ${TESTING_MODE} \
                              --json_file ${json_file_path} \
                              --config_id $i \
                              --backward ${backward} \
                              --use_gpu ${use_gpu} \
                              --repeat $repeat \
                              --allow_adaptive_repeat True"
                        if [[ "${precision}" == "fp16" ]]; then
                            run_cmd="${run_cmd} --convert_to_fp16 True"
                            suffix="_fp16"
                        else
                            suffix=""
                        fi

                        while true
                        do
                            for device_id in ${!DEVICE_TASK_PID_MAP[*]}
                            do
                                task_pid=${DEVICE_TASK_PID_MAP[$device_id]}
                                if [ $task_pid -eq 0 -o -z "$(ps -opid | grep -w $task_pid)" ]
                                then
                                    gpu_id=$device_id
                                    finished_task_pid=$task_pid
                                    break 2
                                fi
                            done
                            sleep 1s
                        done

                        run_start=`date +%s%N`
                        if [ "${OUTPUT_DIR}" != "" ]; then
                            logfile=${OUTPUT_DIR}/${api_name}"_"${i}"-"${framework}"_"${device}"_"${task}"_"${direction}""${suffix}".txt"
                            # Set maxmimum runtime to 10min, or it will be considered
                            #  hanged and will be killed.
                            if [ ${device} = "gpu" ]; then
                                CUDA_VISIBLE_DEVICES="${gpu_id}" timeout 600s ${run_cmd} > $logfile 2>&1 &
                            else
                                CUDA_VISIBLE_DEVICES="" taskset -c ${gpu_id} timeout 600s ${run_cmd} > $logfile 2>&1 &
                            fi
                            task_pid=$!
                        else
                            logfile=""
                            if [ ${device} = "gpu" ]; then
                                CUDA_VISIBLE_DEVICES="${gpu_id}" ${run_cmd} &
                            else
                                CUDA_VISIBLE_DEVICES="" taskset -c ${gpu_id} ${run_cmd} &
                            fi
                            task_pid=$!
                        fi
                        task_pids="${task_pids} ${task_pid}"
                        TASK_PID_RUN_START_MAP[$task_pid]=$run_start
                        DEVICE_TASK_PID_MAP[$gpu_id]=$task_pid
                        TASK_PID_INFO_MAP[$task_pid]="${config_id} ${case_id} ${device} ${backward} ${logfile} ${gpu_id}"
                        TASK_PID_DETAIL_KEY_MAP[$task_pid]=$detail_key
                        [ $finished_task_pid -ne 0 ] && print_finished_task_detail $finished_task_pid $run_start
                    done
                 done
            done
        done
    done
    DETAIL_KEY_TOTAL_TASK_NUM_MAP[$detail_key]=$case_id
    DETAIL_KEY_TASK_PIDS_MAP[$detail_key]="$task_pids"
}

function run_all_cases() {
    local op_info_str=`cat ${OP_LIST_FILE}`
    local op_info_array=(${op_info_str/\\n/ })
    local num_ops=${#op_info_array[*]}

    [ -z "$WORKER_NODE_TOTAL" ] && WORKER_NODE_TOTAL=1
    [ -z "$WORKER_NODE_INDEX" ] && WORKER_NODE_INDEX=0
    local num_ops_each_gpu=$((num_ops+WORKER_NODE_TOTAL-1))
    local num_ops_each_gpu=$((num_ops_each_gpu/WORKER_NODE_TOTAL))
    local config_index_begin=$((WORKER_NODE_INDEX*num_ops_each_gpu))
    local config_index_end=$((config_index_begin+num_ops_each_gpu))
    if [ ${config_index_end} -gt ${num_ops} ]; then
        config_index_end=${num_ops}
    fi

    local config_id=0

    echo "config_index_begin: ${config_index_begin}; config_index_end: ${config_index_end};"
    local line_id=${config_index_begin}
    while [ ${line_id} -lt ${config_index_end} ]; do
        local line=${op_info_array[line_id]}
        local case_name=$(echo $line | cut -d ',' -f1)
        local json_file=$(echo $line | cut -d ',' -f3)
        if [ "$json_file" != "None" ]; then
            local json_file_path=${JSON_CONFIG_DIR}/${json_file}
            local cases_num=$(grep '"op"' ${json_file_path} | wc -l)
        else
            local cases_num=1
            local json_file_path=None
        fi

        if [ -n "$(echo ${case_name} | grep ':')" ]; then
            local case_name_id=${case_name##*:}
        else
            local case_name_id=""
        fi
    
        for((i=0;i<cases_num;i++)); do
            [ -n "${case_name_id}" -a "${case_name_id}" != "${i}" ] && continue
            [ -n "${case_name_id}" ] && line=${line//:${case_name_id}/}
            config_id=$[$config_id+1]
            execute_one_case ${config_id} ${line} ${json_file_path} ${i} ${gpu_id}
        done
        line_id=$((line_id+1))
    done

    while [ ${#DEVICE_TASK_PID_MAP[*]} -gt 0 ]
    do
        for device_id in ${!DEVICE_TASK_PID_MAP[*]}
        do
            task_pid=${DEVICE_TASK_PID_MAP[$device_id]}
            if [ $task_pid -eq 0 ]
            then
                unset DEVICE_TASK_PID_MAP[$device_id]
            elif [ -z "$(ps -opid | grep -w $task_pid)" ]
            then
                print_finished_task_detail $task_pid $(date +%s%N)
                unset DEVICE_TASK_PID_MAP[$device_id]
            fi
        done
        sleep 1s
    done
}

print_arguments $*

cpu_runtime=0
gpu_runtime=0
num_success_cases=0
num_failed_cases=0

run_all_cases

echo "===================================================================="
echo "Summary:"
echo "  ${num_success_cases} successed; ${num_failed_cases} failed"
echo "  GPU runtime: ${gpu_runtime} ms; CPU runtime: ${cpu_runtime} ms"
echo "===================================================================="
