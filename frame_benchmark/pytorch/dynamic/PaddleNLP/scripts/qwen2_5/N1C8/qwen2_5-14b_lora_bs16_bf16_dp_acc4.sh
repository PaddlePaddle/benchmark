model_item=qwen2_5-14b_lora
model_name_or_path=Qwen/Qwen2.5-14B
bs_item=16
fp_item=bf16
run_stage=lora
run_mode=dp_acc4
device_num=N1C8
max_iter=500
num_workers=8

source prepare.sh ${model_name_or_path};
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh ${model_item} ${model_name_or_path} ${bs_item} ${fp_item} ${run_stage} ${run_mode} ${device_num} ${max_iter} ${num_workers} 2>&1;