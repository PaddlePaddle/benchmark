model_item=MobileNetV3_large_x1_0
bs_item=256
fp_item=fp16
run_process_type=MultiP
run_mode=DP
device_num=N1C8
max_epoch=1
num_workers=4
use_compile=true

sed -i '/set\ -xe/d' run_benchmark.sh
bash PrepareEnv.sh;
bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} ${max_epoch} ${num_workers} ${use_compile} 2>&1;
