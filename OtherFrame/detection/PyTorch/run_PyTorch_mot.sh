#!/usr/bin/env bash

ImageName="paddlepaddle/paddle:latest-dev-cuda11.2-cudnn8-gcc82"
docker pull ${ImageName}

export BENCHMARK_ROOT=/workspace # 对应实际地址 benchmark/OtherFrameworks/detection/PyTorch
run_cmd="cd ${BENCHMARK_ROOT}
        bash PrepareEnv_mot.sh
        pip install sklearn  pandas  opencv-python  matplotlib  yacs Cython  cython_bbox  fvcore progress
        cd ${BENCHMARK_ROOT}/models/jde;
        cp ${BENCHMARK_ROOT}/scripts/run_benchmark_mot.sh ./;
        cp ${BENCHMARK_ROOT}/scripts/analysis_log_mot.py ./;
        sed -i '/set\ -xe/d' run_benchmark_mot.sh
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark_mot.sh sp 4 fp32 1 jde;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark_mot.sh mp 4 fp32 1 jde;
        sleep 60;

        cd ${BENCHMARK_ROOT}/models/fairmot/src;
        cp ${BENCHMARK_ROOT}/scripts/run_benchmark_mot.sh ./;
        cp ${BENCHMARK_ROOT}/scripts/analysis_log_mot.py ./;
        sed -i '/set\ -xe/d' run_benchmark_mot.sh
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark_mot.sh sp 6 fp32 1 fairmot;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark_mot.sh mp 6 fp32 1 fairmot;
        sleep 60;
        "
        #CUDA_VISIBLE_DEVICES=0 bash run_benchmark_mot.sh sp 14 fp32 1 jde;
        #sleep 60;
        #CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark_mot.sh mp 14 fp32 1 jde;
        #sleep 60;
        #CUDA_VISIBLE_DEVICES=0 bash run_benchmark_mot.sh sp 18 fp32 1 fairmot;
        #sleep 60;
        #CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark_mot.sh mp 18 fp32 1 fairmot;
        #sleep 60;

nvidia-docker run --name test_torch_mot -i  \
    --net=host \
    --shm-size=64g \
    -v $PWD:/workspace \
    -v /ssd3:/ssd3 \
    -v /ssd2:/ssd2 \
    -e "ALL_PATH=${all_path}" \
    -v "BENCHMARK_ROOT=/workspace" \
    -e "http_proxy=${http_proxy}" \
    -e "https_proxy=${http_proxy}" \
    -e "no_proxy=bcebos.com" \
    ${ImageName}  /bin/bash -c "${run_cmd}"

nvidia-docker stop test_torch_mot
nvidia-docker rm test_torch_mot'
