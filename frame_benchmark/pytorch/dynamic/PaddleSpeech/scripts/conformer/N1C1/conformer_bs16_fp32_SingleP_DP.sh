#!/bin/bash

bash prepare.sh

CUR_DIR=${PWD}
# 1 安装该模型需要的依赖 (如需开启优化策略请注明)
source venv/bin/activate

# 2 拷贝该模型需要数据、预训练模型
cp run_benchmark.sh examples/aishell/s0
cp analysis_log.py examples/aishell/s0
pushd examples/aishell/s0/


mkdir -p exp/log
. path.sh

# 3 批量运行（如不方便批量，1，2需放到单个模型中）

model_item=conformer
bs_item=16
fp_item=fp32
run_process_type=SingleP
run_mode=DP
device_num=N1C1
max_epoch=3

bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} ${max_epoch}  2>&1;

popd

cp examples/aishell/s0/wenet* .
