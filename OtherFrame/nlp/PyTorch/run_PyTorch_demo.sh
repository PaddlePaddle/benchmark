#!/usr/bin/env bash

## 注意，本脚本仅为示例,相关内容请勿更新到此

# 拉镜像
ImageName=  ;
docker pull ${ImageName}

# 启动镜像后测试单个模型
run_cmd="bash PrepareEnv.sh;
        cd /workspace/models/NLP/nlp_modelName/;
        cp /workspace/scripts/NLP/nlp_modelName/preData.sh ./;
        cp /workspace/scripts/NLP/nlp_modelName/run_benchmark.sh ./;
        cp /workspace/scripts/NLP/nlp_modelName/analysis_log.py ./;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 32 fp32 500;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh sp 64 fp16 500;
        "
# 启动镜像
nvidia-docker run --name test_torch -it  \
    --net=host \
    --shm-size=1g \
    -v $PWD:/workspace \
    ${ImageName}  /bin/bash -c "${run_cmd}"


