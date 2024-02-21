#!/usr/bin/env bash

# Test training benchmark for a model.

# Usage: CUDA_VISIBLE_DEVICES=xxx bash run_benchmark.sh ${model_name} ${run_mode} ${fp_item} ${bs_item} ${max_epochs} ${num_workers}

function _set_params(){
    model_item=${1:-"model_item"}   # (必选) 模型 item
    base_batch_size=${2:-"2"}       # (必选) 每张卡上的batch_size
    fp_item=${3:-"fp32"}            # (必选) fp32|fp16
    run_process_type=${4:-"MultiP"} # (必选) 单卡 SingleP|多卡 MultiP
    run_mode=${5:-"DP"}             # (必选) MP模型并行|DP数据并行|PP流水线并行|混合并行DP1-MP1-PP1|DP1-MP4-PP1
    device_num=${6:-"N1C1"}         # (必选) 使用的卡数量，N1C1|N1C8|N4C32 （4机32卡）
    profiling=${PROFILING:-"false"}      # (必选) Profiling  开关，默认关闭，通过全局变量传递
    model_repo="RT-DETR"          # (必选) 模型套件的名字
    ips_unit="samples/sec"         # (必选)速度指标单位
    skip_steps=10                  # (必选)解析日志，跳过模型前几个性能不稳定的step
    keyword="ips:"                 # (必选)解析日志，筛选出性能数据所在行的关键字

    convergence_key=""             # (可选)解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"
    max_epochs=${7:-"1"}                # （可选）需保证模型执行时间在5分钟内，需要修改代码提前中断的直接提PR 合入套件  或是max_epoch
    num_workers=${8:-"2"}             # (可选)

    # Added for distributed training
    multi_nodes=${MULTINODES:-"false"}      # (必选) 多卡训练开关，默认关闭，通过全局变量传递
    node_num=${9:-"2"}                      #（可选） 节点数量
    node_rank=${10:-"0"}                    # (可选)  节点rank
    master_addr=${11:-"127.0.0.1"}       # (可选) 主节点ip地址
    master_port=${12:-"1928"}               # (可选) 主节点端口号

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
    python analysis_log.py "${model_item}" "${log_file}" "${speed_log_file}" "${device_num}" "${base_batch_size}" "${fp_item}"
}

function _train(){
    batch_size="${base_batch_size}"  # 如果模型跑多卡单进程，请在_train函数中计算出多卡需要的bs

    echo "current ${model_name} CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=${device_num}, batch_size=${batch_size}"

    _expected_batch_size=16
    if [ "${batch_size}" -ne "${_expected_batch_size}" ]; then
        echo "batch_size must be ${_expected_batch_size}" 1>&2
        exit 1
    fi

    _expected_run_mode='DP'
    if [ "${run_mode}" != "${_expected_run_mode}" ]; then
        echo "run_mode must be ${_expected_run_mode}" 1>&2
        exit 1
    fi

    _expected_max_epochs=1
    if [ "${max_epochs}" -ne "${_expected_max_epochs}" ]; then
        echo "max_epochs must be ${_expected_max_epochs}" 1>&2
        exit 1
    fi

    _expected_num_workers=16
    if [ "${num_workers}" -ne "${_expected_num_workers}" ]; then
        echo "num_workers must be ${_expected_num_workers}" 1>&2
        exit 1
    fi

    if [ "${fp_item}" = 'fp16' ]; then
        set_amp="--amp"
    else
        set_amp=""
    fi

    train_config="custom_configs/rtdetr/rtdetr_r50vd_4pdx.yml"

    case "${run_process_type}" in
    SingleP) train_cmd="python tools/train.py -c ${train_config}" ;;
    MultiP)
      if [ "${device_num:3}" = '32' ]; then
          train_cmd="python -m torch.distributed.launch --nnodes=${node_num} --node_rank=${node_rank} --master_addr=${master_addr} --master_port=${master_port} --nproc_per_node=8 ./tools/train.py -c ${train_config} ${set_amp}"
      elif [ "${device_num:3}" = '8' ]; then
          train_cmd="python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 ./tools/train.py -c ${train_config} ${set_amp}"
      else
          echo "invalid number of devices" 1>&2
          exit 1
      fi  ;;
    *) echo "choose run_mode(SingleP or MultiP)"; exit 1;
    esac

#   以下为通用执行命令，无特殊可不用修改
    timeout 5m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
    else
        echo -e "${model_name}, SUCCESS"
    fi
    #kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
    if [ ${run_process_type} = "MultiP" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}

_set_params "$@"
export frame_version=`python -c "import torch;print(torch.__version__)"`
echo "---------frame_version is torch ${frame_version}"
echo "---------model_branch is ${model_branch}"
echo "---------model_commit is ${model_commit}"

job_bt=`date '+%Y%m%d%H%M%S'`
_train
job_et=`date '+%Y%m%d%H%M%S'`
export model_run_time=$((${job_et}-${job_bt}))
_analysis_log
