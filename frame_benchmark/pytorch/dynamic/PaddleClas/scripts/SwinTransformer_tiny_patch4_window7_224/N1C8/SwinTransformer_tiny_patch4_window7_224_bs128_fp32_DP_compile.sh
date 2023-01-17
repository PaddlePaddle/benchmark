model_item=SwinTransformer_tiny_patch4_window7_224
bs_item=128
fp_item=fp32
run_process_type=MultiP
run_mode=DP
device_num=N1C8
max_epoch=1
num_workers=4
use_compile=true

sed -i '/set\ -xe/d' run_benchmark.sh
bash PrepareEnv.sh;
bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} ${max_epoch} ${num_workers} 2>&1;
