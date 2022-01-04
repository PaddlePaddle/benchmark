#!/usr/bin/env bash
set -xe
# 参数说明
function _set_params(){
    run_mode=${1:-"sp"}
    batch_size=${2:-"64"}
    fp_item=${3:-"fp32"}        # fp32|fp16
    max_iter=${4:-"10"}       # 如果需要修改代码提前中断
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}

    model_name="ocr_PSENet"
    mission_name="OCR"            # 模型所属任务名称，具体可参考scripts/config.ini  必填）
    direction_id=0                 # 任务所属方向，0：CV，1：NLP，2：Rec。     (必填)
    ips_unit="samples/sec"

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    log_file=${run_log_path}/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}
    index_log_file=${run_log_path}/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}_speed
}
function _train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
    # 防止checkpoints冲突
    rm -rf checkpoints/
    # 如需开启特殊优化flag、参数请注明

    train_cmd="config/psenet/psenet_r50_ic15_736.py --cfg-options data.batch_size=${batch_size} train_cfg.epoch=${max_iter} model.backbone.pretrained=False"
    case ${run_mode} in
    sp) train_cmd="python3.7 train.py "${train_cmd} ;;
    mp)
        train_cmd="python3.7 train.py "${train_cmd} ;;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac
    
    timeout 15m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
        export job_fail_flag=1
    else
        echo -e "${model_name}, SUCCESS"
        export job_fail_flag=0
    fi

    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}

function _analysis_log(){
    analysis_cmd="python3.7 analysis_log.py --filename ${log_file}  --mission_name ${model_name} --run_mode ${run_mode} --direction_id 0 --keyword 'ips:' --base_batch_size ${batch_size} --skip_steps 1 --gpu_num ${num_gpu_devices}  --index 1  --model_mode=-1  --ips_unit=samples/sec"
    eval $analysis_cmd
}

function _kill_process(){
    kill -9 `ps -ef|grep 'python3.7'|awk '{print $2}'`
}
_set_params $@
_train
_analysis_log
_kill_process
