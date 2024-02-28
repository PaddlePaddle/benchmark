model_item='nanodet_plus_m_1.5x_4pdx'
bs_item=64
fp_item='fp16'
run_process_type='SingleP'
run_mode='DP'
device_num='N1C1'
max_epochs=1
num_workers=6
repeats=24

bash PrepareEnv.sh
bash repeat_data.sh "${repeats}"
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
