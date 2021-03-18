#!bin/bash
set -xe
if [[ $# -lt 1 ]]; then
    echo "running job dict is {1: speed, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 1|3|6 sp|mp model_name(HRnet|deeplabv3) 600(max_iter)"
    exit
fi

function _set_params(){
    index=$1
    base_batch_size=2
    run_mode=${2:-"sp"} # Use sp for single GPU and mp for multiple GPU.
    model_name=${3}
    max_iter=${4:-"200"}

    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}

    mission_name="图像分割"
    direction_id=0
    skip_steps=5
    keyword="ips:"
    model_mode=-1
    ips_unit="images/s"

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}
    batch_size=`expr ${base_batch_size} \* ${num_gpu_devices}`  # 静态图未总的bs

    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/${model_name}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}
}

function _train(){
    export PYTHONPATH=$(pwd):{PYTHONPATH}
    export FLAGS_cudnn_exhaustive_search=1
    if [ ${model_name} = "HRnet" ]; then
        config="configs/hrnetw18_cityscapes_1024x512_215.yaml"
    elif [ ${model_name} = "deeplabv3" ]; then
        config="configs/deeplabv3p_resnet50_vd_cityscapes.yaml"
    else
        echo "------------------>model_name should be HRnet or deeplabv3!"
        exit 1
    fi
    grep -q "#To address max_iter" pdseg/train.py
    if [ $? -eq 0 ]; then
        echo "----------already addressed max_iter"
    else
        sed -i '/data_loader.start()/a\        max_step_id = '${max_iter}' #To address max_iter' pdseg/train.py
        sed -i '/reader_cost_averager.record(time.time() - batch_start)/i\                if step == max_step_id: return' pdseg/train.py
    fi

    train_cmd="--cfg=${config}
               --use_gpu
               BATCH_SIZE ${batch_size}
               DATALOADER.NUM_WORKERS 2
               SOLVER.NUM_EPOCHS 1"

    if [ ${run_mode} = "sp" ]; then
        train_cmd="python -u pdseg/train.py "${train_cmd}
    else
        rm -rf ./mylog
        train_cmd="python -m paddle.distributed.launch  --gpus=$CUDA_VISIBLE_DEVICES --log_dir ./mylog pdseg/train.py "${train_cmd}
        log_parse_file="mylog/workerlog.0"
    fi

    echo "#################################${model_name}"
    timeout 15m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
        export job_fail_flag=1
    else
        echo -e "${model_name}, SUCCESS"
        export job_fail_flag=0
    fi
    echo "#################################${model_name}"
    if [ ${run_mode} != "sp"  -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_run
