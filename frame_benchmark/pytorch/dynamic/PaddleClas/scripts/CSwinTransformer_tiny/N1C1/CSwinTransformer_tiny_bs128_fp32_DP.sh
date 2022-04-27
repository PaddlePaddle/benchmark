model_item=CSwinTransformer_tiny
bs_item=128
fp_item=fp32
device_num=N1C1
run_mode=DP
max_epoch=100
num_workers=4

bash PrepareEnv.sh;
bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} ${max_epoch} ${num_workers} 2>&1;
