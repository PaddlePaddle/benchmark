cp models/wenet/examples/aishell/s0/conformer_mp_bs16_fp32_8 .

python analysis.py \
    --filename "conformer_mp_bs16_fp32_8" \
    --keyword "avg_ips:" \
    --base_batch_size 16 \
    --model_name "Conformer" \
    --mission_name "eight gpu" \
    --run_mode "mp" \
    --ips_unit "sent./sec" \
    --gpu_num 8 \
    --use_num 480 \
    --separator " " \

