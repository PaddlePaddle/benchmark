#!/usr/bin/env bash

ImageName="registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7";
docker pull ${ImageName}
export BENCHMARK_ROOT=/workspace # 对应实际地址 benchmark/OtherFrameworks/detection/PyTorch
run_cmd="cd ${BENCHMARK_ROOT}
        bash PrepareEnv_mot.sh
        cd ${BENCHMARK_ROOT}/models/jde;
        cp ${BENCHMARK_ROOT}/scripts/run_benchmark_mot.sh ./;
        cp ${BENCHMARK_ROOT}/scripts/analysis_log_mot.py ./;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark_mot.sh sp 1 fp32 1 jde;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark_mot.sh mp 1 fp32 1 jde;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark_mot.sh sp 8 fp32 1 jde;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark_mot.sh mp 8 fp32 1 jde;
        sleep 60;
        cd ${BENCHMARK_ROOT}/models/fairmot;
        cp ${BENCHMARK_ROOT}/scripts/run_benchmark_mot.sh ./;
        cp ${BENCHMARK_ROOT}/scripts/analysis_log_mot.py ./;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark_mot.sh sp 2 fp32 1 fairmot;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark_mot.sh mp 2 fp32 1 fairmot;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark_mot.sh sp 8 fp32 1 fairmot;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark_mot.sh mp 8 fp32 1 fairmot;
        sleep 60;
        "

nvidia-docker run --name test_torch_mot -it  \
    --net=host \
    --shm-size=64g \
    -v $PWD:/workspace \
    ${ImageName}  /bin/bash -c "${run_cmd}"

nvidia-docker stop test_torch_mot
nvidia-docker rm test_torch_mot'
