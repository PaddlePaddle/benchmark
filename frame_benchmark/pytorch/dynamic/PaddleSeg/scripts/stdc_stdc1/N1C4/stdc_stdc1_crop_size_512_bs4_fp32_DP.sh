model_item="stdc_stdc1"
bs_item=4
fp_item=fp32
run_mode=DP
device_num=N1C4
max_iter=400
num_workers=24
crop_size=512

bash prepare.sh;
bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} ${max_iter} ${num_workers} ${crop_size} 2>&1;
