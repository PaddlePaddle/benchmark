model_item=petrv2
bs_item=1
fp_item=fp32
run_process_type=MultiP
run_mode=DP
device_num=N4C32
max_iter=-1
num_workers=32

node_num=${PADDLE_TRAINERS_NUM}
node_rank=${PADDLE_TRAINER_ID}
master_addr=${POD_0_IP}
master_port=14333

sed -i '/set\ -xe/d' run_benchmark.sh
bash PrepareEnv.sh ${model_item};
bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} ${max_iter} ${num_workers} ${node_num} ${node_rank} ${master_addr} ${master_port} 2>&1;
