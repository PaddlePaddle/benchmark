#! /bin/bash

# Usage:
#   bash main_control.sh json_config_dir output_dir gpu_id cpu|gpu|both speed|accuracy|both"

export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

OP_BENCHMARK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../" && pwd )"
DEPLOY_DIR="${OP_BENCHMARK_ROOT}/deploy"
TEST_DIR="${OP_BENCHMARK_ROOT}/tests"
export PYTHONPATH=${OP_BENCHMARK_ROOT}:${PYTHONPATH}

JSON_CONFIG_DIR=${1:-"${TEST_DIR}/examples"}

OUTPUT_DIR=${2:-""}
if [ ! -d ${OUTPUT_DIR} ]; then
    mkdir -p ${OUTPUT_DIR}
fi

GPU_ID=${3:-"0"}

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

function print_detail_status() {
    config_id=$1
    case_id=$2
    device=$3
    backward=$4
    logfile=$5
    runtime=$6
    return_status=$7

    if [ ${backward} = "False" ]; then
        backward_shorten="F"
    else # backward="True"
        backward_shorten="T"
    fi
    if  [ ${return_status} -eq 0 ]; then
        run_status="SUCCESS"
    elif [ ${runtime} -ge 600000 ]; then
        run_status="**TIMEOUT**"
    else
        run_status="**FAILED**"
    fi
    print_str="device=${device}, backward=${backward_shorten}, ${logfile}, time=${runtime} ms"
    print_str_length=${#print_str}
    timestamp=`date +"%Y-%m-%d %T"`
    if [ ${print_str_length} -lt 80 ]; then
        printf "  [%d-%d][%s] %-80s ...... %s\n" ${config_id} ${case_id} "${timestamp}" "${print_str}" "${run_status}"
    elif [ ${print_str_length} -lt 90 ]; then
        printf "  [%d-%d][%s] %-90s ...... %s\n" ${config_id} ${case_id} "${timestamp}" "${print_str}" "${run_status}"
    elif [ ${print_str_length} -lt 100 ]; then
        printf "  [%d-%d][%s] %-100s ...... %s\n" ${config_id} ${case_id} "${timestamp}" "${print_str}" "${run_status}"
    else
        printf "  [%d-%d][%s] %-120s ...... %s\n" ${config_id} ${case_id} "${timestamp}" "${print_str}" "${run_status}"
    fi
}

if [ ${OUTPUT_DIR} != "" ]; then
    api_info_file=${OUTPUT_DIR}/api_info.txt
else
    api_info_file=api_info.txt
fi
python ${DEPLOY_DIR}/collect_api_info.py --info_file ${api_info_file}
return_status=$?
if [ ${return_status} -ne 0 ] || [ ! -f "${api_info_file}" ]; then
    api_info_file=${DEPLOY_DIR}/api_info.txt
fi

config_id=0
cpu_runtime=0
gpu_runtime=0
num_success_cases=0
num_failed_cases=0

for line in `cat $api_info_file`
do
    api_name=$(echo $line| cut -d',' -f1)
    name=$(echo $line| cut -d',' -f2)
    json_file=$(echo $line| cut -d',' -f3)
    has_backward=$(echo $line| cut -d',' -f4)

    direction_set=("forward" "backward")
    if [ ${has_backward} = False ]; then  
        direction_set=("forward")
    fi

    if [ "$json_file" != "None" ]
    then
        json_file_path=${JSON_CONFIG_DIR}/${json_file}
        cases_num=$(grep '"op"' ${json_file_path} |wc -l)
    else
        cases_num=1
        json_file_path=None
    fi

    for((i=0;i<cases_num;i++));
    do
        config_id=$[$config_id+1]
        echo "[${config_id}]: api_name=${api_name}, name=${name}, json_file=${json_file_path}, num_configs=${cases_num}, json_id=${i}"
        case_id=0
        # device: gpu, cpu
        for device in ${DEVICE_SET[@]};
        do 
            if [ ${device} = "gpu" ]; then
                export CUDA_VISIBLE_DEVICES="${GPU_ID}"
                use_gpu="True"
                repeat=1000
            else
                export CUDA_VISIBLE_DEVICES=""
                use_gpu="False"
                repeat=100
            fi
            # task: speed, accuracy
            for task in "${TASK_SET[@]}";
            do 
                framwork_set=("paddle" "tensorflow")
                if [ ${task} = "accuracy" ]; then
                    framwork_set=("paddle")
                fi
                # framework: paddle, tensorflow
                for framework in "${framwork_set[@]}";
                do 
                    # direction: forward, backward
                    for direction in "${direction_set[@]}"; 
                    do
                        if [ ${direction} = "forward" ]; then
                            backward="False"
                        else
                            backward="True"
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
                            # Set maxmimum runtime to 10min, or it will be considered hanged and will be killed.
                            timeout 600s ${run_cmd} > $logfile 2>&1
                            return_status=$?
                        else
                            logfile=""
                            ${run_cmd}
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
                        print_detail_status ${config_id} ${case_id} "${device}" "${backward}" "${logfile}" ${runtime} ${return_status}
                     done
                done
            done
        done
        echo ""
    done
done

echo "===================================================================="
echo "Summary:"
echo "  ${num_success_cases} successed; ${num_failed_cases} failed"
echo "  GPU runtime: ${gpu_runtime} ms; CPU runtime: ${cpu_runtime} ms"
echo "===================================================================="
