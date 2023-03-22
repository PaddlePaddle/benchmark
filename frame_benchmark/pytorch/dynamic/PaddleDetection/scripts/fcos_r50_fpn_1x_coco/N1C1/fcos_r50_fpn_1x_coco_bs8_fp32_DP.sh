model_item=fcos_r50_fpn_1x_coco
bs_item=8
fp_item=fp32
run_process_type=SingleP
run_mode=DP
device_num=N1C1
max_epochs=5
num_workers=2

sed -i '/set\ -xe/d' run_benchmark.sh
bash PrepareEnv.sh;
bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} ${max_epochs} ${num_workers} 2>&1;
