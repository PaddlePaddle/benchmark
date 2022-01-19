#!/usr/bin/env bash
set -xe
# 运行示例：CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} 500 ${model_mode}
# 参数说明
function _set_params(){
    run_mode=${1:-"sp"}          # 单卡sp|多卡mp
    batch_size=${2:-"1"}
    fp_item=${3:-"fp32"}        # fp32|fp16
    model_name=${4:-"model_name"}
    pushd ../../
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}/scripts/logs  # TRAIN_LOG_DIR 后续QA设置该参数
    popd
    echo ${run_log_path}

    mission_name="视频分类"
    direction_id=0

    if [ ! -d $run_log_path/train_log ];then
        mkdir -p $run_log_path/train_log
    fi

    if [ ! -d $run_log_path/index ];then
        mkdir -p $run_log_path/index
    fi

#   以下不用修改   
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    # log_file=${run_log_path}/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}
    log_file=${run_log_path}/train_log/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}
    res_log_file=${run_log_path}/index/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}_speed
}

function _analysis_log(){
    python3.7 analysis_log.py \
                --filename ${log_file} \
                --jsonname ${res_log_file} \
                --keyword "avg_ips:" \
                --model_name video_TimeSformer_bs${batch_size}_${fp_item} \
                --mission_name ${mission_name} \
                --direction_id ${direction_id} \
                --run_mode ${run_mode} \
                --gpu_num ${num_gpu_devices} \
                --index 1
    cp ${res_log_file} ${BENCHMARK_ROOT}/
}

function _train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
    total_bs=`expr $batch_size \* $num_gpu_devices`
    case ${run_mode} in
    sp)
        if [ ${fp_item} == 'fp32' ]; then
            train_cmd="python3.7 tools/run_net.py --cfg configs/UCF101/${model_name}_divST_8x32_224.yaml NUM_GPUS ${num_gpu_devices} TRAIN.BATCH_SIZE ${total_bs}  SOLVER.MAX_EPOCH 1"
            # echo excute: ${train_cmd}
        elif [ ${fp_item} == 'fp16' ]; then
            train_cmd="python3.7 -u main.py --amp -c configs/recognition/${model_name}/${model_name}_ucf101_videos_benchmark_bs${batch_size}.yaml" 
            # echo excute: ${train_cmd}
        else
            echo "choose fp_item(fp32 or fp16)"
            exit 1
        fi;;
    mp)
        if [ ${fp_item} == 'fp32' ]; then
            train_cmd="python3.7 tools/run_net.py --cfg configs/UCF101/${model_name}_divST_8x32_224.yaml NUM_GPUS ${num_gpu_devices} TRAIN.BATCH_SIZE ${total_bs} SOLVER.MAX_EPOCH 1"
            # echo excute: ${train_cmd}
            log_parse_file="mylog/workerlog.0"
        elif [ ${fp_item} == 'fp16' ]; then
            train_cmd="python3.7 -B -m paddle.distributed.launch --gpus=$CUDA_VISIBLE_DEVICES --log_dir=./mylog main.py --amp -c configs/recognition/${model_name}/${model_name}_ucf101_videos_benchmark_bs${batch_size}_mp.yaml"
            # echo excute: ${train_cmd}
            log_parse_file="mylog/workerlog.0"
        else
            echo "choose fp_item(fp32 or fp16)"
            exit 1
        fi;;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac
# 以下不用修改
    # 删除之前的log文件
    if [[ -f ${log_file} ]]; then
        rm -rf ${log_file}
    fi
    timeout 15m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
        export job_fail_flag=1
    else
        echo -e "${model_name}, SUCCESS"
        export job_fail_flag=0
    fi

    _analysis_log

    kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
 
    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}
 
_set_params $@
_train
