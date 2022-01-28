model_item=basicvsr
bs_item=2
fp_item=fp32
run_process_type=SingleP
run_mode=DP
device_num=N1C1

sed -i '/set\ -xe/d' benchmark/run_benchmark.sh
bash benchmark/prepare.sh ${model_item};
bash benchmark/run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} 2>&1;
sleep 10;
export PROFILING=true
bash benchmark/run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} 2>&1;
