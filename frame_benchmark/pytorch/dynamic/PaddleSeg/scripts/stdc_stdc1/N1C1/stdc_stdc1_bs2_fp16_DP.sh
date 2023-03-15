model_item="stdc_stdc1"
bs_item=2
fp_item=fp16
run_mode=DP
device_num=N1C1
max_iter=400
num_workers=8

bash prepare.sh;
bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} ${max_iter} ${num_workers} 2>&1;
