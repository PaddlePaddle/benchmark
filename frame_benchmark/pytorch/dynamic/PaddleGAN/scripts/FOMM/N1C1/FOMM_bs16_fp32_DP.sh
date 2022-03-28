model_item=FOMM
bs_item=16
fp_item=fp32
run_process_type=SingleP
run_mode=DP
device_num=N1C1

sed -i '/set\ -xe/d' run_benchmark.sh
bash prepare.sh ${model_item};
bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} 2>&1;