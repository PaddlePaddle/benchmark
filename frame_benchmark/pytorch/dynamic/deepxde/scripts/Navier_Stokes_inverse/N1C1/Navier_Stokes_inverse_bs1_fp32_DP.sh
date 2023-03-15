model_item=Navier_Stokes_inverse
bs_item=1
fp_item=fp32
run_mode=DP
device_num=N1C1
#prepare
bash prepare.sh
#run
bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} 2>&1;
sleep 10;
