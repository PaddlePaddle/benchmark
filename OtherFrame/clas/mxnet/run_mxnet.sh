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
	bash run.sh;
	mv clas* /workspace/
        "
# 启动镜像
nvidia-docker run --name test_mxnet -it  \
    --net=host \
    --shm-size=64g \
    -v $PWD:/workspace \
    ${ImageName}  /bin/bash -c "${run_cmd}"
