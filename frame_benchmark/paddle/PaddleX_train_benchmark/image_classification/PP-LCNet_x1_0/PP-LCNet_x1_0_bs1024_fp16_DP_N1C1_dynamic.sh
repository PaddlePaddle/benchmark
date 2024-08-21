# 该配置执行脚本包括timeout、多卡日志拷贝等个性化设置
#bash PaddleX/train_benchmark/image_classification/prepare.sh

export model_item=PP-LCNet_x1_0
export model_type=dynamic
export batch_size=1024
export fp_item=fp16
export run_mode=DP
export device_num=N1C1
export model_name=PP-LCNet_x1_0_bs1024_fp16_DP
export job_name=PaddleX_image_classification_PP-LCNet_x1_0_bs1024_fp16_DP_N1C1
export skip_steps=10
export keyword="ips: "
export convergence_key="loss: "
export speed_unit="samples/s"
export log_file=${TRAIN_LOG_DIR:-$PWD}/${job_name}_log
export speed_log_file=${LOG_PATH_INDEX_DIR:-$PWD}/${job_name}_speed
export distributed_train_logs=${MODEL_REPO_ROOT:-$PWD}/output/${job_name}/distributed_train_logs

train_cmd="python main.py -c paddlex/configs/image_classification/PP-LCNet_x1_0.yaml -o Global.mode=train -o Global.dataset_dir=dataset/ILSVRC2012_train_benchmark -o Train.epochs_iters=1 -o Train.log_interval=1 -o Train.num_classes=1000 -o Benchmark.shuffle=False -o Benchmark.shared_memory=True -o Benchmark.print_mem_info=True -o Benchmark.seed=1234 -o Benchmark.do_eval=False -o Benchmark.disable_deamon=True -o Benchmark.env.FLAGS_eager_delete_tensor_gb=0.0 -o Benchmark.env.FLAGS_fraction_of_gpu_memory_to_use=0.98 -o Benchmark.env.FLAGS_conv_workspace_size_limit=4096 -o Global.model=PP-LCNet_x1_0 -o Train.batch_size=1024 -o Benchmark.dali=True -o Benchmark.num_workers=16 -o Benchmark.amp=O2 -o Global.device=gpu:${CUDA_VISIBLE_DEVICES} -o Global.output=output/${job_name}"

echo $train_cmd
timeout 5m $train_cmd > $log_file 2>&1
echo $train_cmd >> $log_file
