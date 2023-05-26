model_item=latent_diffusion_model
bs_item=8
fp_item=fp16
run_process_type=SingleP
run_mode=DP
device_num=N1C1
max_iter=500
num_workers=1

sed -i '/set\ -xe/d' run_benchmark.sh
bash prepare.sh;
CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} ${max_iter} ${num_workers} 2>&1;
