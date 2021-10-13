#!/usr/bin/env bash
# 拉镜像
ImageName="registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7";
docker pull ${ImageName}
# 启动镜像后测试单个模型
run_cmd="
        cd /workspace/models/gluon-cv/;
        cp /workspace/scripts/*.sh ./;
        cp /workspace/scripts/analysis_log.py ./;
        bash PrepareEnv.sh
        bash PrepareData.sh;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 64 fp32 500 mobilenet1.0;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 64 fp32 500 mobilenet1.0;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 768 fp32 500 mobilenet1.0;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 768 fp32 500 mobilenet1.0;
        mv clas* /workspace/
        "
# 启动镜像
nvidia-docker run --name test_mxnet -it  \
    --net=host \
    --shm-size=64g \
    -v $PWD:/workspace \
    ${ImageName}  /bin/bash -c "${run_cmd}"
