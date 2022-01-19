#!/bin/bash

CUR_DIR=${PWD}

# 1 安装该模型需要的依赖 (如需开启优化策略请注明)
source venv/bin/activate

# 2 拷贝该模型需要数据、预训练模型
pushd models/wenet/examples/aishell/s0/

mkdir -p exp/log
. path.sh

bash run.sh --data $PWD --stop_stage 3

# 3 批量运行（如不方便批量，1，2需放到单个模型中）

model_mode_list=(conformer)
fp_item_list=(fp32)
bs_item_list=(16)
for model_mode in ${model_mode_list[@]}; do
      for fp_item in ${fp_item_list[@]}; do
          for bs_item in ${bs_item_list[@]}
            do
            rm exp -rf
            echo "index is speed, 1gpus, begin, ${model_name}"
            run_mode=sp
            CUDA_VISIBLE_DEVICES=0 bash ${CUR_DIR}/run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} 500 ${model_mode}     #  (5min)
            sleep 60
            rm exp -rf
            echo "index is speed, 8gpus, run_mode is multi_process, begin, ${model_name}"
            run_mode=mp
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ${CUR_DIR}/run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} 500 ${model_mode}
            sleep 60
            done
      done
done

popd # aishell/s1

mkdir -p log
bash run_analysis_sp.sh > log/log_sp.out
bash run_analysis_mp.sh > log/log_mp.out
