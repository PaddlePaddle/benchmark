model_item="facebook-llama-13b_lora"  
base_batch_size=8  
fp_item="fp16"    
run_mode="DP1"    
device_num="N1C1"
model_name_or_path="huggyllama/llama-13b"
lora="1"
max_length=2048
dataset_name_or_path="llm_benchmark_en"
learning_rate="3e-04"
gradient_checkpointing="1"
num_train_epochs=2
bash prepare.sh
bash llama-13b.sh
export CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ${model_item} ${base_batch_size} ${fp_item} ${run_mode} ${device_num} ${model_name_or_path} ${lora} ${max_length} ${dataset_name_or_path} ${learning_rate} ${gradient_checkpointing} ${gradient_accumulation_steps} ${num_train_epochs} 2>&1;