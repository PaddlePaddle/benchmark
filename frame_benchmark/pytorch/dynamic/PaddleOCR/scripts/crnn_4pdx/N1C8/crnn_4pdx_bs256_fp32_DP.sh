model_item='crnn_4pdx'
bs_item=256
fp_item='fp32'
run_process_type='MultiP'
run_mode='DP'
device_num='N1C8'
max_epochs=1
num_workers=16

bash PrepareEnv.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh \
    "${model_item}" \
    "${bs_item}" \
    "${fp_item}" \
    "${run_process_type}" \
    "${run_mode}" \
    "${device_num}" \
    "${max_epochs}" \
    "${num_workers}" \
    2>&1
