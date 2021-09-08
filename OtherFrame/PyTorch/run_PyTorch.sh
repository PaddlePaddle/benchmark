#!/usr/bin/env bash

# 拉镜像
ImageName=  ;
docker pull ${ImageName}

# 启动镜像后测试单个模型
run_cmd="bash PrepareEnv.sh;
        cd **;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 32 fp32 500;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh sp 64 fp16 500;
        "
NV_VISIBLE_DEVICES=${2:-"all"}

# 启动镜像
nvidia-docker run --name test_bert_torch -it  \
    --net=host \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $PWD:/workspace \
    ${ImageName}  /bin/bash ${run_cmd}


