model_item=det_r50_db_v2_0
#max_token
bs_item=8
fp_item=fp16
run_process_type=MultiP
run_mode=DP
device_num=N4C32
max_iter=5
num_workers=1

sed -i '/set\ -xe/d' run_benchmark.sh
bash prepare.sh;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} ${max_iter} ${num_workers} 2>&1;
