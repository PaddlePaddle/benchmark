model_item=ResNet152
bs_item=32
fp_item=fp32
run_process_type=MultiP
run_mode=DP
device_num=N4C32
max_epoch=1
num_workers=4

node_num=${PADDLE_TRAINERS_NUM}
node_rank=${PADDLE_TRAINER_ID}
master_addr=${POD_0_IP}
master_port=14233

sed -i '/set\ -xe/d' run_benchmark.sh
bash PrepareEnv.sh;

# mmcv-full wheel in PrepareEnv.sh takes long time to install
# use this script to sync between different machines.
cat <<EOF >tmp.py
import torch
flag = True
while flag:
    try:
        torch.distributed.init_process_group(backend="nccl")
        flag = False
    except:
        pass
EOF
python -m torch.distributed.run --nnodes=${node_num} --node_rank=${node_rank} --master_addr=${master_addr} --master_port=${master_port} --nproc_per_node=8 tmp.py

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} ${max_epoch} ${num_workers} ${node_num} ${node_rank} ${master_addr} ${master_port} 2>&1;