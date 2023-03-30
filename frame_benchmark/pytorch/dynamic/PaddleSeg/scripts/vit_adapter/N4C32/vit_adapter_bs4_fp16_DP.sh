model_item="vit_adapter"
bs_item=4
fp_item=fp16
run_mode=DP
device_num=N4C32
max_iter=400
num_workers=24
train_config=configs/ade20k/upernet_augreg_adapter_tiny_512_160k_ade20k.py

bash prepare.sh;
bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} ${max_iter} ${num_workers} ${train_config} 2>&1;
