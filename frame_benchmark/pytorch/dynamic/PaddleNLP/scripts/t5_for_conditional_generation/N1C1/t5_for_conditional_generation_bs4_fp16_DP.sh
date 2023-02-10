model_item=t5_for_conditional_generation
bs_item=4
fp_item=fp16
run_process_type=SingleP
run_mode=DP
device_num=N1C1

sed -i '/set\ -xe/d' run_benchmark.sh
bash PrepareEnv.sh;
CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} 2>&1;