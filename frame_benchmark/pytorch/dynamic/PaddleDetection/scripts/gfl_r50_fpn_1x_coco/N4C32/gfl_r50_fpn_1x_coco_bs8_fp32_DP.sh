model_item=gfl_r50_fpn_1x_coco
bs_item=8
fp_item=fp32
run_process_type=MultiP
run_mode=DP
device_num=N4C32
max_epochs=1
num_workers=2

sed -i '/set\ -xe/d' run_benchmark.sh
bash PrepareEnv.sh;

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

bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} ${max_epochs} ${num_workers} 2>&1;
