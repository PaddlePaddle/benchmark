model_item=transformer_big
#max_token
bs_item=4096
fp_item=fp16
run_process_type=MultiP
run_mode=DP
device_num=N4C32
max_iter=500
num_workers=8

node_num=${PADDLE_TRAINERS_NUM}
node_rank=${PADDLE_TRAINER_ID}
master_addr=${POD_0_IP}
master_port=14233

sed -i '/set\ -xe/d' run_benchmark.sh
bash PrepareEnv.sh;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} ${max_iter} ${num_workers} ${node_num} ${node_rank} ${master_addr} ${master_port} 2>&1;