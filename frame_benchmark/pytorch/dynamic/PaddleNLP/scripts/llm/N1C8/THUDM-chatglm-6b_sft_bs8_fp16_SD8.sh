model_item="THUDM-chatglm-6b_sft"  
base_batch_size=8    
fp_item="fp16"    
run_mode="SD8"    
device_num="N1C8"
model_name_or_path="THUDM/chatglm-6b"
lora="false"
max_length=2048
dataset_name_or_path="llm_benchmark_zh"
learning_rate="3e-05"
gradient_checkpointing="true"
gradient_accumulation_steps=2
num_train_epochs=2
bash prepare.sh
bash chatglm-6b.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash run_benchmark.sh ${model_item} ${base_batch_size} ${fp_item} ${run_mode} ${device_num} ${model_name_or_path} ${lora} ${max_length} ${dataset_name_or_path} ${learning_rate} ${gradient_checkpointing} ${gradient_accumulation_steps} ${num_train_epochs} 2>&1;