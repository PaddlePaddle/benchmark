# 该配置执行脚本包括timeout、多卡日志拷贝等个性化设置
#bash PaddleX/train_benchmark/text_detection/prepare.sh

export model_item=PP-OCRv4_mobile_det
export model_type=dynamic
export batch_size=8
export fp_item=fp32
export run_mode=DP
export device_num=N1C1
export model_name=PP-OCRv4_mobile_det_bs8_fp32_DP
export job_name=PaddleX_text_detection_PP-OCRv4_mobile_det_bs8_fp32_DP_N1C1
export skip_steps=10
export keyword="ips: "
export convergence_key="loss: "
export speed_unit="samples/s"
export log_file=${TRAIN_LOG_DIR:-$PWD}/${job_name}_log
export speed_log_file=${LOG_PATH_INDEX_DIR:-$PWD}/${job_name}_speed
export distributed_train_logs=${MODEL_REPO_ROOT:-$PWD}/output/${job_name}/distributed_train_logs

train_cmd="python main.py -c paddlex/configs/text_detection/PP-OCRv4_mobile_det.yaml -o Global.device=gpu:${CUDA_VISIBLE_DEVICES} -o Global.mode=train -o Global.dataset_dir=dataset/ocr_det_dataset_train_benchmark -o Global.output=output/${job_name} -o Train.batch_size=8 -o Train.log_interval=1 -o Train.epochs_iters=1"

echo $train_cmd
timeout 5m $train_cmd > $log_file 2>&1
echo $train_cmd >> $log_file
