model_item=yolov5_l_300e_coco
bs_item=128
fp_item=fp16
run_process_type=MultiP
run_mode=DP
device_num=N1C8
max_epochs=1
num_workers=4

sed -i '/set\ -xe/d' run_benchmark.sh
bash PrepareEnv.sh;
export http_proxy=${HTTP_PRO}
export https_proxy=${HTTPS_PRO}
bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} ${max_epochs} ${num_workers} 2>&1;
unset http_proxy
unset https_proxy
