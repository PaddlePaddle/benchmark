model_item=dino_r50_4scale_1x_coco
bs_item=16
fp_item=fp32
run_process_type=MultiP
run_mode=DP
device_num=N1C8
max_iter=780
num_workers=4

sed -i '/set\ -xe/d' run_benchmark.sh
bash PrepareEnv.sh;
export http_proxy=${HTTP_PRO}
export https_proxy=${HTTPS_PRO}
bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} ${max_iter} ${num_workers} 2>&1;
unset http_proxy
unset https_proxy
