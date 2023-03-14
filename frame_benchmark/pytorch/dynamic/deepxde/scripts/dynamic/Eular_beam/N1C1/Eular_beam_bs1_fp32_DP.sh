model_item=Eular_beam
bs_item=1
fp_item=fp32
run_mode=DP
device_num=N1C1
#prepare
bash test_tipc/dynamic/${model_item}/benchmark_common/prepare.sh
#run
bash test_tipc/dynamic/${model_item}/benchmark_common/run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} 2>&1;
sleep 10;
