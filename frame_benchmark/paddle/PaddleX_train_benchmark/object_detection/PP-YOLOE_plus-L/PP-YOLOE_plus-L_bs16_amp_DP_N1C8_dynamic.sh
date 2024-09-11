# 该配置执行脚本包括timeout、多卡日志拷贝等个性化设置
#bash PaddleX/train_benchmark/object_detection/prepare.sh

export model_item=PP-YOLOE_plus-L
export model_type=dynamic
export batch_size=16
export fp_item=amp
export run_mode=DP
export device_num=N1C8
export model_name=PP-YOLOE_plus-L_bs16_amp_DP
export job_name=PaddleX_object_detection_PP-YOLOE_plus-L_bs16_amp_DP_N1C8
export skip_steps=10
export keyword="ips: "
export convergence_key="loss: "
export speed_unit="samples/s"
export log_file=${TRAIN_LOG_DIR:-$PWD}/${job_name}_log
export speed_log_file=${LOG_PATH_INDEX_DIR:-$PWD}/${job_name}_speed
export distributed_train_logs=${MODEL_REPO_ROOT:-$PWD}/output/${job_name}/distributed_train_logs

train_cmd="python main.py -c paddlex/configs/object_detection/PP-YOLOE_plus-L.yaml -o Global.mode=train -o Global.dataset_dir=dataset/coco_train_benchmark -o Train.epochs_iters=1 -o Train.log_interval=1 -o Train.num_classes=80 -o Benchmark.shuffle=False -o Benchmark.shared_memory=True -o Benchmark.print_mem_info=True -o Benchmark.do_eval=False -o Benchmark.disable_deamon=True -o Global.model=PP-YOLOE_plus-L -o Train.batch_size=16 -o Benchmark.num_workers=4 -o Benchmark.amp=O1 -o Global.device=gpu:${CUDA_VISIBLE_DEVICES} -o Global.output=output/${job_name}"

echo $train_cmd
timeout 5m $train_cmd > $log_file 2>&1
echo $train_cmd >> $log_file
