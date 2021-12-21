#!/usr/bin/env bash
set -xe
# 运行示例：CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} ${model_mode} ${config_path}
# 参数说明
function _set_params(){
    run_mode=${1:-"sp"}          # 单卡sp|多卡mp
    batch_size=${2:-"64"}
    fp_item=${3:-"fp32"}        # fp32|fp16
    model_item=${4:-"model_item"}
    config_path=${5:-"config_path"}
    run_log_path="${TRAIN_LOG_DIR:-$(pwd)}"  # TRAIN_LOG_DIR 后续QA设置该参
 
#   以下不用修改   
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    log_file=${run_log_path}/${model_item}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}.log
    model_name=HRNet_W48_C # model_name 在analysis 里面拼接fp以及bs，构成json格式
}
function _train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"

    sed -i "s/BATCH_SIZE_PER_GPU: 32/BATCH_SIZE_PER_GPU: $batch_size/g" experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
    train_cmd="python tools/train.py --cfg experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml"
    case ${run_mode} in
    sp) sed -ie 's/GPUS: (0,1,2,3)/GPUS: (0,)/g' experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml;;
    mp)
	sed -ie 's/GPUS: (0,1,2,3)/GPUS: (0,1,2,3,4,5,6,7)/g' experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
	sed -i 's/WORKERS: 4/WORKERS: 32/g' experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml;;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac

# 以下不用修改
    timeout 15m ${train_cmd}  > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
        export job_fail_flag=1
    else
        echo -e "${model_name}, SUCCESS"
        export job_fail_flag=0
    fi
    kill -9 `ps -ef|grep 'python'|awk '{print $2}'`

    case ${run_mode} in
    sp) sed -ie 's/GPUS: (0,)/GPUS: (0,1,2,3)/g' experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml;;
    mp)
	sed -ie 's/GPUS: (0,1,2,3,4,5,6,7)/GPUS: (0,1,2,3)/g' experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
	sed -i 's/WORKERS: 32/WORKERS: 4/g' experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml;;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac

    sed -i "s/BATCH_SIZE_PER_GPU: $batch_size/BATCH_SIZE_PER_GPU: 32/g" experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml

}

_set_params $@
rm -rf output
sed -i 's/view/reshape/g' lib/core/evaluate.py
sed -i 's/PRINT_FREQ: 1000/PRINT_FREQ: 10/g' experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
source ${ROOT_DIR}/scripts/run_model.sh
_run
python analysis_log.py -d output -m ${model_name} -b ${batch_size} -n ${num_gpu_devices}
