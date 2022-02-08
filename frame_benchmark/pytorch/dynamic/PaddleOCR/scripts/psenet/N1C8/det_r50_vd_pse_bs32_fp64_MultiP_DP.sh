model_item=det_r50_vd_pse
#max_token
bs_item=32
fp_item=fp64
run_process_type=MultiP
run_mode=DP
device_num=N1C8
max_iter=10
num_workers=1

sed -i '/set\ -xe/d' run_benchmark.sh
bash PrepareEnv.sh;
bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} ${max_iter} ${num_workers} 2>&1;
