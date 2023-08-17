#!/bin/bash
#prepare
cd ${BENCHMARK_ROOT}
if [ -d "logs" ];then rm -rf logs
fi
mkdir logs
cd logs
mkdir static
mkdir dynamic
pip install --upgrade pip
pip install opencv-python==4.6.0.66
pip install tqdm
pip install paddlenlp
export FLAGS_call_stack_level=2
# run models
cd ${BENCHMARK_ROOT}/scripts/benchmark_ci
#model_list='ResNet50_bs32_dygraph ResNet50_bs32 bert_base_seqlen128_fp32_bs32 transformer_base_bs4096_amp_fp16 yolov3_bs8 TSM_bs16 deeplabv3_bs4_fp32 CycleGAN_bs1 mask_rcnn_bs1 PPOCR_mobile_2_bs8 seq2seq_bs128'
model_list='ResNet50_bs32_dygraph deeplabv3_bs4_fp32 bert_base_seqlen128_fp32_bs32 yolov3_bs8 ppyolov2_bs6 ResNet50_pure_fp16_bs64'
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
if [ -f "errorcode.txt" ];then rm -rf errorcode.txt
fi
echo success >>log.txt
python analysis.py --log_path=${BENCHMARK_ROOT}/logs/static --standard_path=${BENCHMARK_ROOT}/scripts/benchmark_ci/standard_value/static --loss_threshold=1 --paddle_dev=False
python analysis.py --log_path=${BENCHMARK_ROOT}/logs/dynamic --standard_path=${BENCHMARK_ROOT}/scripts/benchmark_ci/standard_value/dynamic --loss_threshold=1 --paddle_dev=False
#if the fluctuations is larger than threshold, then rerun in paddle develop for result judging to avoid fluctuations caused by xiaolvyun machines.
if [ -f "rerun_model.txt" ];then
    echo -e "rerun model in paddle develop start!"
    #install paddle develop
    #avoid import error in paddle develop
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/python3.7.0/lib/python3.7/site-packages/paddle/fluid/../libs/
    cd /workspace/Paddle
    pip uninstall -y paddlepaddle_gpu
    pip install build/dev_whl/paddlepaddle_gpu*.whl
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
        python analysis.py --log_path=${BENCHMARK_ROOT}/logs/static_pr --standard_path=${BENCHMARK_ROOT}/scripts/benchmark_ci/standard_value/static --paddle_dev=True 
    fi
    if [ -d "${BENCHMARK_ROOT}/logs/dynamic" ];then
        python analysis_dev.py --log_path=${BENCHMARK_ROOT}/logs/dynamic --standard_path=${BENCHMARK_ROOT}/scripts/benchmark_ci/standard_value/dynamic
        python analysis.py --log_path=${BENCHMARK_ROOT}/logs/dynamic_pr --standard_path=${BENCHMARK_ROOT}/scripts/benchmark_ci/standard_value/dynamic --paddle_dev=True
    fi
fi
errorcode='0'
if [ -f "errorcode.txt" ];then
    errorcode=`cat errorcode.txt`
fi
if [[ ${errorcode} != '0' ]];then
    errorcode=`expr $errorcode + 20`
fi
if [[ -z `cat log.txt | grep success` ]];then
    echo -e "model benchmark ci job failed!"
    echo -e "See https://github.com/PaddlePaddle/Paddle/wiki/PR-CI-Model-benchmark-Manual for details."
    echo -e "Or you can apply for QA xiegegege's approval to pass this PR." 
    exit ${errorcode}
else
    echo -e "model benchmark ci job success!"
fi
