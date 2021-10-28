#!/usr/bin/env bash

ImageName="registry.baidubce.com/paddlepaddle/paddle:2.0.2-gpu-cuda10.1-cudnn7";
docker pull ${ImageName}

# 启动镜像后测试单个模型
run_cmd="cp /workspace/scripts/PrepareEnv.sh ./
         bash PrepareEnv.sh;
         cd /workspace/models/PSENet/;
         pip3 install -r requirement.txt;
         cp /workspace/scripts/PSENet/preData.sh ./;
         cp /workspace/scripts/PSENet/run_benchmark.sh ./;
         cp /workspace/scripts/PSENet/analysis_log.py ./;
         CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 8 fp32 2;
         CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 64 fp32 2;
        "

nvidia-docker run --name test_torch_ocr -it  \
    --net=host \
    --shm-size=128g \
    -v $PWD:/workspace \
    ${ImageName}  /bin/bash -c "${run_cmd}"

nvidia-docker stop test_torch_ocr
nvidia-docker rm test_torch_ocr 