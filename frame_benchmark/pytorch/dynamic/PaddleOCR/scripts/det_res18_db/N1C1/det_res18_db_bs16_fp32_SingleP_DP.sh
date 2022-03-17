model_item=det_res18_db
#max_token
bs_item=16
fp_item=fp32
run_process_type=SingleP
run_mode=DP
device_num=N1C1
max_iter=2
num_workers=1

sed -i '/set\ -xe/d' run_benchmark.sh
bash PrepareEnv.sh;
CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} ${max_iter} ${num_workers} 2>&1;
