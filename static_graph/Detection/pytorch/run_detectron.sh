#!bin/bash
set -xe
if [[ $# -lt 3 ]]; then
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 speed|mem|maxbs model_name sp /ssd1/ljh/logs"
    echo "model_name can be:"
    echo "  mask_rcnn_fpn_resnet, mask_rcnn_fpn_resnext, retinanet_rcnn_fpn, cascade_fpn_rcnn"
    exit
fi

function _set_params(){
    index=$1                         # 速度(speed)|显存占用(mem)|单卡最大支持batch_size(maxbs)(必填)
    base_batch_size=2                # 单卡的batch_size，如果固定的，可以写死（必填）
    model_name=$2                    # 模型名字如。如果是固定的，可以写死，如果需要其他参数可以参考bert实现（必填）
    run_log_path=${4:-$(pwd)}        # 训练保存的日志目录（必填）
    run_mode=${3:-"sp"}
    skip_steps=2                     # 解析日志，有些模型前几个step耗时长，需要跳过(必填)
    keyword="json_stats"             # 解析日志，筛选出数据所在行的关键字(必填)
    separator=" "                    # 解析日志，数据所在行的分隔符(必填)

    model_mode=0                     # 解析日志，s/step -> samples/s (必填)
    range=-1
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_parse_file=${log_file}

}

function _train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"

    WORK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    if [[ ${model_name} = "cascade_fpn_rcnn" ]]; then
        export PYTHONPATH=${WORK_ROOT}/Detectron-Cascade-RCNN:$PYTHONPATH
        cp -f ${WORK_ROOT}/src/dataset_catalog.py ${WORK_ROOT}/Detectron-Cascade-RCNN/detectron/datasets/dataset_catalog.py
        # Update some files of Detectron-Cascade-RCNN with that of Detectron
        cp -f ${WORK_ROOT}/Detectron/detectron/utils/io.py ${WORK_ROOT}/Detectron-Cascade-RCNN/detectron/utils/io.py
        cp -f ${WORK_ROOT}/Detectron/detectron/utils/env.py ${WORK_ROOT}/Detectron-Cascade-RCNN/detectron/utils/env.py
        cd ${WORK_ROOT}/Detectron-Cascade-RCNN
        python setup.py install --user
        make
        cd ${WORK_ROOT}
        train_cmd="python -u Detectron-Cascade-RCNN/tools/train_net.py \
                        --cfg configs/e2e_cascade_rcnn_R-50-FPN_1x.yaml \
		                    OUTPUT_DIR ./output"
        position=64
    else
        echo "model_name must be mask_rcnn_fpn_resnet | mask_rcnn_fpn_resnext | retinanet_rcnn_fpn"
        exit 1
    fi

    python -c 'import torch; print(torch.__version__)'
    ${train_cmd} > ${log_file} 2>&1 &
    train_pid=$!
    sleep 600
    kill -9 `ps -ef|grep python |awk '{print $2}'`
}

if [ "${BENCHMAKR_ROOT}" == "" ]; then
    export BENCHMARK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../../.." && pwd )"
fi
source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_run
