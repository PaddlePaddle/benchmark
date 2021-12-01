#!/usr/bin/env bash

ImageName="registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7";
docker pull ${ImageName}
export BENCHMARK_ROOT=/workspace 

run_cmd="bash PrepareEnv.sh
        cd ${BENCHMARK_ROOT}/models/Parakeet/PWGAN/;
        cp ${BENCHMARK_ROOT}/scripts/Parakeet/PWGAN/preData.sh ./;
        cp ${BENCHMARK_ROOT}/scripts/Parakeet/PWGAN/run_benchmark.sh ./;
        cp ${BENCHMARK_ROOT}/scripts/Parakeet/PWGAN/analysis_log.py ./;
        bash preData.sh
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh Parakeet_PWGAN sp fp32 6 100 3;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh Parakeet_PWGAN mp fp32 6 100 3;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh Parakeet_PWGAN sp fp32 26 100 3;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh Parakeet_PWGAN mp fp32 26 100 3;
        "

# 启动镜像
nvidia-docker run --name test_torch -it  \
    --net=host \
    --shm-size=1g \
    -v $PWD:/workspace \
    ${ImageName}  /bin/bash -c "${run_cmd}"
