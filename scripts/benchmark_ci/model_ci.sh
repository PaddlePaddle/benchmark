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
pip install paddlenlp
#run models
cd ${BENCHMARK_ROOT}/scripts/benchmark_ci
model_list='ResNet50_bs32_dygraph ResNet50_bs32 bert_base_seqlen128_fp32_bs32'
source run_models.sh
for model in ${model_list}
do
${model}
done
#analysis log
cd ${BENCHMARK_ROOT}/scripts/benchmark_ci
if [ -f "rerun_model.txt" ];then rm -rf rerun_model.txt
fi
if [ -f "log.txt" ];then rm -rf log.txt
fi
echo success >>log.txt
python analysis.py --log_path=${BENCHMARK_ROOT}/logs/static --standard_path=${BENCHMARK_ROOT}/scripts/benchmark_ci/standard_value/static --threshold=0.05 --paddle_dev=False
python analysis.py --log_path=${BENCHMARK_ROOT}/logs/dynamic --standard_path=${BENCHMARK_ROOT}/scripts/benchmark_ci/standard_value/dynamic --threshold=0.05 --paddle_dev=False
#if the fluctuations is larger than threshold, then rerun in paddle develop for result judging to avoid fluctuations caused by xiaolvyun machines.
if [ -f "rerun_model.txt" ];then
    echo -e "rerun model in paddle develop start!"
    #compile paddle develop
    cd /workspace/Paddle
    git checkout develop
    bash -x paddle/scripts/paddle_build.sh build
    [ $? -ne 0 ] && echo "build paddle develop failed." && exit 1
    pip uninstall -y paddlepaddle_gpu
    pip install build/python/dist/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl
    [ $? -ne 0 ] && echo "install paddle failed." && exit 1
    #running model in paddle develop
    mv ${BENCHMARK_ROOT}/logs/static ${BENCHMARK_ROOT}/logs/static_pr
    mv ${BENCHMARK_ROOT}/logs/dynamic ${BENCHMARK_ROOT}/logs/dynamic_pr
    mkdir ${BENCHMARK_ROOT}/logs/static
    mkdir ${BENCHMARK_ROOT}/logs/dynamic
    cd ${BENCHMARK_ROOT}/scripts/benchmark_ci
    model_list=`cat rerun_model.txt`
    echo -e "rerun model list is ${model_list}"
    source run_models.sh
    for model in ${model_list}
    do
        ${model}
    done
    cd ${BENCHMARK_ROOT}/scripts/benchmark_ci
    if [ -d "${BENCHMARK_ROOT}/logs/static" ];then
        #change standard_value as paddle dev result
        python analysis_dev.py --log_path=${BENCHMARK_ROOT}/logs/static --standard_path=${BENCHMARK_ROOT}/scripts/benchmark_ci/standard_value/static 
        #compare the result of pr and paddle dev 
        python analysis.py --log_path=${BENCHMARK_ROOT}/logs/static_pr --standard_path=${BENCHMARK_ROOT}/scripts/benchmark_ci/standard_value/static --threshold=0.05 --paddle_dev=True 
    fi
    if [ -d "${BENCHMARK_ROOT}/logs/dynamic" ];then
        python analysis_dev.py --log_path=${BENCHMARK_ROOT}/logs/dynamic --standard_path=${BENCHMARK_ROOT}/scripts/benchmark_ci/standard_value/dynamic
        python analysis.py --log_path=${BENCHMARK_ROOT}/logs/dynamic_pr --standard_path=${BENCHMARK_ROOT}/scripts/benchmark_ci/standard_value/dynamic --threshold=0.05  --paddle_dev=True
    fi
fi
if [[ -z `cat log.txt | grep success` ]];then
    echo -e "model benchmark ci job failed!"
    echo -e "See https://github.com/PaddlePaddle/Paddle/wiki/PR-CI-Model-benchmark-Manual for details."
    echo -e "Or you can apply for one QA(xiegegege(Recommend), hysunflower) approval to pass this PR." 
    exit 1
else
    echo -e "model benchmark ci job success!"
fi
