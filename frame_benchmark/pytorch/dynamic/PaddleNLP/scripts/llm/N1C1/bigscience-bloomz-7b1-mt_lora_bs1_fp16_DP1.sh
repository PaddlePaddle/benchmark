model_item="bigscience-bloomz-7b1-mt_lora"  
base_batch_size=1
fp_item="fp16"    
run_mode="DP1"    
device_num="N1C1"
model_name_or_path="bigscience/bloomz-7b1-mt"
lora="1"
max_length=2048
dataset_name_or_path="llm_benchmark_zh"
learning_rate="3e-04"
gradient_checkpointing="1"
gradient_accumulation_steps=32
num_train_epochs=2
bash prepare.sh
bash bloom-7b1.sh
export CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ${model_item} ${base_batch_size} ${fp_item} ${run_mode} ${device_num} ${model_name_or_path} ${lora} ${max_length} ${dataset_name_or_path} ${learning_rate} ${gradient_checkpointing} ${gradient_accumulation_steps} ${num_train_epochs} 2>&1;