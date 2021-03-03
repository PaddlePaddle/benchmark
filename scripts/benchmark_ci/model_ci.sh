#!/bin/bash
#prepare
cd ${BENCHMARK_ROOT}
if [ -d "logs" ];then rm -rf logs
fi
mkdir logs
cd logs
mkdir static
mkdir dynamic
pip install opencv-python==4.2.0.32
pip install tqdm
#run models
cd ${BENCHMARK_ROOT}/scripts/benchmark_ci
source run_models.sh
#analysis log
cd ${BENCHMARK_ROOT}/scripts/benchmark_ci
export err_flag=false
python analysis.py --log_path=${BENCHMARK_ROOT}/logs/static --standard_path=${BENCHMARK_ROOT}/scripts/benchmark_ci/standard_value/static --threshold=0.05 
python analysis.py --log_path=${BENCHMARK_ROOT}/logs/dynamic --standard_path=${BENCHMARK_ROOT}/scripts/benchmark_ci/standard_value/dynamic --threshold=0.05 
if [ "${err_flag}" = true ];then
    echo -e "model_benchmark ci job failed!" 
    exit 1
else
    echo -e "model_benchmark ci job success!"
fi


