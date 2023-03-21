model_item="maskformer"
bs_item=2
fp_item=fp16
run_mode=DP
device_num=N1C4
max_iter=400
num_workers=8
train_config="configs/ade20k-150/swin/maskformer_swin_tiny_bs16_160k.yaml"

bash prepare.sh;
bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} ${max_iter} ${num_workers} ${train_config} 2>&1;