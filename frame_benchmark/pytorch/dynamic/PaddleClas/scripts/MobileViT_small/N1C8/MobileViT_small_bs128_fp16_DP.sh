model_item=mobilevit_small
bs_item=128
fp_item=fp16
device_num=N1C8
run_mode=DP
max_epoch=100
num_workers=4

bash PrepareEnv.sh;
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} ${max_epoch} ${num_workers} 2>&1;
