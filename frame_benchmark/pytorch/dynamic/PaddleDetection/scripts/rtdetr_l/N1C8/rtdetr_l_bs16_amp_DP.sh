model_item='rtdetr_l'
bs_item=16
fp_item='amp'
run_process_type='MultiP'
run_mode='DP'
device_num='N1C8'
max_epochs=1
num_workers=16
repeats=96

bash PrepareEnv.sh
bash run_benchmark.sh \
    "${model_item}" \
    "${bs_item}" \
    "${fp_item}" \
    "${run_process_type}" \
    "${run_mode}" \
    "${device_num}" \
    "${max_epochs}" \
    "${num_workers}" \
    2>&1
