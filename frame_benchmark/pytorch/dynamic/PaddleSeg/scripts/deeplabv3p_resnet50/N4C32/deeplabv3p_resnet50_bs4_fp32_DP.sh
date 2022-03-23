model_item=deeplabv3p_resnet50
bs_item=4
fp_item=fp32
run_mode=DP
device_num=N4C32
max_iter=500
num_workers=5

bash prepare.sh;
bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} ${max_iter} ${num_workers} 2>&1;
