output_path="op_summary_GPU_2.0_1.8_1016.xlsx"

op_result_dir1="/benchmark/op_GPU_logs/1016/Paddle/2.0"
op_result_dir2="/benchmark/op_GPU_logs/1016/Paddle/1.8"

url_prefix1="http://10.255.129.33:8999/1016/Paddle/2.0/"
url_prefix2="http://10.255.129.33:8999/1016/Paddle/1.8/"

python summary.py \
     ${op_result_dir1} \
     ${op_result_dir2} \
    --op_frequency_path "op_frequency.txt" \
    --dump_to_excel True \
    --dump_to_mysql False \
    --output_path ${output_path} \
    --url_prefix1 ${url_prefix1} \
    --url_prefix2 ${url_prefix2} \
#    --output_path ${output_path}
