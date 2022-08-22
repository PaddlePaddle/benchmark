#!/usr/bin/env bash

# Test training benchmark for a model.

# Usage: CUDA_VISIBLE_DEVICES=xxx bash run_benchmark.sh ${model_name} ${run_mode} ${fp_item} ${bs_item} ${max_iter} ${num_workers}

function _set_params(){
    model_item=${1:-"model_item"}
    base_batch_size=${2:-"2"}
    fp_item=${3:-"fp32"}        # fp32 or fp16
    run_mode=${4:-"DP"}
    device_num=${5:-"N1C1"}
    profiling=${PROFILING:-"false"}
    model_repo="mmedit"
    speed_unit="samples/sec"
    skip_steps=10
    max_iter=${6:-"100"}                # （可选）需保证模型执行时间在5分钟内，需要修改代码提前中断的直接提PR 合入套件  或是max_epoch
    num_workers=${7:-"3"}               # (可选)

    # Added for distributed training
    node_num=${8:-"2"}                      #（可选） 节点数量
    node_rank=${9:-"0"}                    # (可选)  节点rank
    master_addr=${10:-"127.0.0.1"}       # (可选) 主节点ip地址
    master_port=${11:-"1928"}               # (可选) 主节点端口号
    # Added for distributed training

    #   以下为通用拼接log路径，无特殊可不用修改
    model_name=${model_item}_bs${base_batch_size}_${fp_item}_${run_mode}  # (必填) 切格式不要改动,与平台页面展示对齐
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}  # （必填） TRAIN_LOG_DIR  benchmark框架设置该参数为全局变量
    profiling_log_path=${PROFILING_LOG_DIR:-$(pwd)}  # （必填） PROFILING_LOG_DIR benchmark框架设置该参数为全局变量
    speed_log_path=${LOG_PATH_INDEX_DIR:-$(pwd)}
    train_log_file=${run_log_path}/${model_repo}_${model_name}_${device_num}_log
    profiling_log_file=${profiling_log_path}/${model_repo}_${model_name}_${device_num}_profiling
    speed_log_file=${speed_log_path}/${model_repo}_${model_name}_${device_num}_speed
    if [ ${profiling} = "true" ];then
            add_options="profiler_options=\"batch_range=[50, 60]; profile_path=model.profile\""
            log_file=${profiling_log_file}
        else
            add_options=""
            log_file=${train_log_file}
    fi
}

function _analysis_log(){
    python analysis_log.py ${model_item} ${base_batch_size} ${log_file} ${speed_log_file} ${device_num}
}

function _train(){
    batch_size=${base_batch_size}
    echo "current ${model_name} CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=${device_num}, batch_size=${batch_size}"
    train_config="mmedi_benchmark_configs/${model_name}.py"
    train_options="--no-validate "

    case ${device_num} in
    N1C1) train_cmd="./tools/dist_train.sh ${train_config} 1 ${train_options}" ;;
    N1C8)
        case ${model_name} in
        basicvsr_mp_bs2|basicvsr_mp_bs4) train_cmd="./tools/dist_train.sh ${train_config} 4 ${train_options}" ;;
        *) train_cmd="./tools/dist_train.sh ${train_config} 8 ${train_options}"
        esac
        ;;
    N4C32)
        train_cmd="python -m torch.distributed.launch --nnodes=${node_num} --node_rank=${node_rank} \
                                --nproc_per_node=8 --master_port=${master_port} --master_addr=${master_addr} \
                                tools/train.py ${train_config} --launcher pytorch ${train_options}" ;;
    *) echo "choose device_num(N1C1, N1C8 or N4C32)"; exit 1;
    esac

    timeout 5m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
    else
        echo -e "${model_name}, SUCCESS"
    fi
    if [ ${device_num} != "N1C1" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
    #kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
}

_set_params $@
export frame_version=`python -c "import torch;print(torch.__version__)"`
echo "---------frame_version is torch ${frame_version}"
echo "---------model_branch is ${model_branch}"
echo "---------model_commit is ${model_commit}"
job_bt=`date '+%Y%m%d%H%M%S'`
_train
job_et=`date '+%Y%m%d%H%M%S'`
export model_run_time=$((${job_et}-${job_bt}))
_analysis_log
