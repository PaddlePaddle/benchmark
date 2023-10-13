model_item=segformer_b0
bs_item=4
fp_item=fp16
run_mode=DP
device_num=N1C1
max_iter=500
num_workers=8
crop_size=512

bash prepare.sh;
bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} ${max_iter} ${num_workers} ${crop_size} 2>&1;
