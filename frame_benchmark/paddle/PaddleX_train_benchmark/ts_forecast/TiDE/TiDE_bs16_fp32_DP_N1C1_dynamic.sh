# 该配置执行脚本包括timeout、多卡日志拷贝等个性化设置
#bash PaddleX/train_benchmark/ts_forecast/prepare.sh

export model_item=TiDE
export model_type=dynamic
export batch_size=16
export fp_item=fp32
export run_mode=DP
export device_num=N1C1
export model_name=TiDE_bs16_fp32_DP
export job_name=PaddleX_ts_forecast_TiDE_bs16_fp32_DP_N1C1
export skip_steps=5
export keyword="ips: "
export convergence_key="loss: "
export speed_unit="sequences/sec"
export log_file=${TRAIN_LOG_DIR:-$PWD}/${job_name}_log
export speed_log_file=${LOG_PATH_INDEX_DIR:-$PWD}/${job_name}_speed
export distributed_train_logs=${MODEL_REPO_ROOT:-$PWD}/output/${job_name}/distributed_train_logs

train_cmd="python main.py -c paddlex/configs/modules/ts_forecast/TiDE.yaml -o Global.mode=train -o Global.dataset_dir=dataset/ts_forecast_train_benchmark -o Train.epochs_iters=5 -o Train.patience=40 -o Benchmark.print_mem_info=True -o Benchmark.disable_deamon=True -o Benchmark.env.FLAGS_eager_delete_tensor_gb=0.0 -o Benchmark.env.FLAGS_fraction_of_gpu_memory_to_use=0.98 -o Benchmark.env.FLAGS_conv_workspace_size_limit=4096 -o Global.model=TiDE -o Train.batch_size=16 -o Global.device=gpu:${CUDA_VISIBLE_DEVICES} -o Global.output=output/${job_name} -o Train.uniform_output_enabled=False"

echo $train_cmd
timeout 15m $train_cmd > $log_file 2>&1
echo $train_cmd >> $log_file
