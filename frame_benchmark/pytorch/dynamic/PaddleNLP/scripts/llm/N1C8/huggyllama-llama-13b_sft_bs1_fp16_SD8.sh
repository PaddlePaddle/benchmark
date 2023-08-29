model_item="huggyllama-llama-13b_sft"  
base_batch_size=1    
fp_item="fp16"    
run_mode="SD8"    
device_num="N1C8"
model_name_or_path="huggyllama/llama-13b"
lora="false"
max_length=2048
dataset_name_or_path="llm_benchmark_en"
learning_rate="3e-05"
gradient_checkpointing="true"

bash prepare.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash run_benchmark.sh ${model_item} ${base_batch_size} ${fp_item} ${run_mode} ${device_num} ${model_name_or_path} ${lora} ${max_length} ${dataset_name_or_path} ${learning_rate} ${gradient_checkpointing} 2>&1;