model_item=TimeSformer
bs_item=14
fp_item=fp32
run_process_type=MultiP
run_mode=DP
device_num=N1C1

bash prepare.sh;
CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} 2>&1;
