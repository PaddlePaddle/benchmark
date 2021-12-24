#!/usr/bin/env bash

#http_proxy=http://172.19.57.45:3128

ImageName="registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7";
docker pull ${ImageName}
export BENCHMARK_ROOT=/workspace 

run_cmd="cd /workspace 
        bash PrepareEnv.sh
        cd ${BENCHMARK_ROOT}/models/Parakeet/PWGAN/;
        cp ${BENCHMARK_ROOT}/scripts/Parakeet/PWGAN/preData.sh ./;
        cp ${BENCHMARK_ROOT}/scripts/Parakeet/PWGAN/run_benchmark.sh ./;
        cp ${BENCHMARK_ROOT}/scripts/Parakeet/PWGAN/analysis_log.py ./;
        sed -i '/set\ -xe/d' run_benchmark.sh
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh Parakeet_pwgan sp fp32 6 100 3;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh Parakeet_pwgan mp fp32 6 100 3;
        "
        #bash preData.sh
#        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh Parakeet_pwgan sp fp32 26 100 3;  # 模型名改成小写 和benchmark paddle 一致 否则数据无法前端正常显示
#        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh Parakeet_pwgan mp fp32 26 100 3;

# 启动镜像
nvidia-docker run --name test_torch_speech -i  \
    --net=host \
    --shm-size=1g \
    -v $PWD:/workspace \
    -v /ssd3:/ssd3 \
    -v /ssd2:/ssd2 \
    -v "BENCHMARK_ROOT=/workspace" \
    -e "http_proxy=${http_proxy}" \
    -e "https_proxy=${http_proxy}" \
    -e "no_proxy=bcebos.com" \
    ${ImageName}  /bin/bash -c "${run_cmd}"

docker stop test_torch_speech
docker rm test_torch_speech
