model_item=GhostNet_x1_0
bs_item=512
fp_item=fp32
run_process_type=SingleP
run_mode=DP
device_num=N1C1
max_epoch=1
num_workers=12

sed -i '/set\ -xe/d' run_benchmark.sh
bash PrepareEnv.sh;
bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} ${max_epoch} ${num_workers} 2>&1;
