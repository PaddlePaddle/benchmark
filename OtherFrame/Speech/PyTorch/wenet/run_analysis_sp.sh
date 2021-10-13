cp models/wenet/examples/aishell/s0/conformer_sp_bs16_fp32_1 .

python analysis.py \
    --filename "conformer_sp_bs16_fp32_1" \
    --keyword "avg_ips:" \
    --base_batch_size 16 \
    --model_name "Conformer" \
    --mission_name "one gpu" \
    --run_mode "sp" \
    --ips_unit "sent./sec" \
    --gpu_num 1 \
    --use_num 60 \
    --separator " " \
    --direction_id "1"
