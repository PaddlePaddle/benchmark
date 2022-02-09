#!/bin/bash

CUR_DIR=${PWD}
# 1 安装该模型需要的依赖 (如需开启优化策略请注明)
source venv/bin/activate

# 2 拷贝该模型需要数据、预训练模型

# 3 批量运行（如不方便批量，1，2需放到单个模型中）

model_item=pwgan
bs_item=26
fp_item=fp32
run_process_type=MultiP
run_mode=DP
device_num=N1C8
max_iter=100

bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} ${max_iter}  2>&1;
