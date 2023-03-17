model_item=gpt_for_sequence_classification
bs_item=8
fp_item=fp16
run_process_type=MultiP
run_mode=DP
device_num=N1C8

sed -i '/set\ -xe/d' run_benchmark.sh
bash PrepareEnv.sh;
unset CUDA_VISIBLE_DEVICES
bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} 2>&1;