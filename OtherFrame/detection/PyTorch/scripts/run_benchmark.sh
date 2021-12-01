#!/usr/bin/env bash
set -xe
# Usage：CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ${run_mode} ${batch_size} ${fp_item} ${max_epoch} ${model_name}

function _set_params(){
    run_mode=${1:-"sp"}            # sp|mp
    batch_size=${2:-"2"}           #
    fp_item=${3:-"fp32"}           # fp32|fp16
    max_epoch=${4:-"1"}            #
    model_name=${5:-"model_name"}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}  # TRAIN_LOG_DIR

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    log_file=${run_log_path}/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}
    res_log_file=${run_log_path}/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}_speed
}

function _analysis_log(){
    python analysis_log.py \
                --filename ${log_file} \
                --jsonname ${res_log_file} \
                --keyword "time:" \
                --model_name detection_${model_name}_bs${batch_size}_${fp_item} \
                --run_mode ${run_mode} \
                --gpu_num ${num_gpu_devices} \
                --batch_size ${batch_size}
    cp ${res_log_file} /workspace
}

function _train(){
    echo "Train ${model_name} on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"

    # parse model_name
    case ${model_name} in
        faster_rcnn) model_yml="configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py" ;;
        fcos) model_yml="configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py" ;;
        deformable_detr) model_yml="configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py" ;;
        gfl) model_yml="configs/gfl/gfl_r50_fpn_1x_coco.py" ;;
        solov2) model_yml="configs/solov2/solov2_r50_fpn_8gpu_1x.py" ;;
        hrnet_w32_keypoint) model_yml="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py" ;;
        higherhrnet_w32) model_yml="configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w32_coco_512x512.py" ;;
        *) echo "Undefined model_name"; exit 1;
    esac

    set_batch_size="data.samples_per_gpu=${batch_size}"
    set_max_epoch="runner.max_epochs=${max_epoch}"
    set_max_epoch_pose="total_epochs=${max_epoch}"
    set_log_iter="log_config.interval=1"
    if [ ${fp_item} = "fp16" ]; then
        set_fp_item="fp16.loss_scale=512."
    else
        set_fp_item=" "
    fi

    if [ ${model_name} = "solov2" ]; then
        train_cmd=""
    else
        train_cmd="--no-validate \
            --cfg-options ${set_max_epoch_pose} ${set_fp_item} ${set_batch_size} ${set_max_epoch} ${set_log_iter}"
    fi
    case ${run_mode} in
        sp) train_cmd="python -u tools/train.py ${model_yml} ${train_cmd}";;
        mp) train_cmd="bash ./tools/dist_train.sh ${model_yml} 8 ${train_cmd}";;
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

    _analysis_log

    kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
}

_set_params $@
_train
