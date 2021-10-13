#!/usr/bin/env bash
# 拉镜像
ImageName="registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7";
docker pull ${ImageName}

# 启动镜像后测试HRNet
run_cmd="
        cd /workspace/models/HRNet-Image-Classification/;
        cp /workspace/scripts/HRNet-Image-Classification_scripts/*.sh ./;
        cp /workspace/scripts/HRNet-Image-Classification_scripts/analysis_log.py ./;
	bash PrepareEnv.sh 
	bash PrepareData.sh;
	bash run.sh;
	mv clas* /workspace/
        "

# 启动镜像后测试Twins
run_cmd="
        cd /workspace/models/Twins;
        cp /workspace/scripts/Twins_scripts/*.sh ./;
        cp /workspace/scripts/Twins_scripts/analysis_log.py ./;
        bash PrepareEnv.sh 
        bash PrepareData.sh;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 64 fp32 500 alt_gvt_base;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 176 fp32 500 alt_gvt_base;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 64 fp32 500 alt_gvt_base;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 176 fp32 500 alt_gvt_base;
        mv clas* /workspace/
        "

# 启动镜像后测试MobileNetV2, MobileNetV3, ShuffleNetV2, SwinTransformer
run_cmd="
        cd /workspace/models/mmclassification;
        cp /workspace/scripts/mmclassification_scripts/*.sh ./;
        cp /workspace/scripts/mmclassification_scripts/analysis_log.py ./;
	bash PrepareEnv.sh 
	bash PrepareData.sh;
	bash run_all.sh;
	mv clas* /workspace/
        "


# 启动镜像
nvidia-docker run --name test_pytorch -it  \
    --net=host \
    --shm-size=64g \
    -v $PWD:/workspace \
    ${ImageName}  /bin/bash -c "${run_cmd}"

