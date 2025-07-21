model_item=dpa2
bs_item=1
fp_item=fp64
run_process_type='SingleP'
run_mode=DP
device_num=N1C1

sed -i '/set\ -xe/d' run_benchmark.sh
bash PrepareEnv.sh;
bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} 2>&1;