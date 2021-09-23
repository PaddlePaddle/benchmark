#!/usr/bin/env bash

ImageName="registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7";
docker pull ${ImageName}

run_cmd="cp /workspace/PrepareEnv.sh ./;
         bash PrepareEnv.sh;
         cd /home/mmsegmentation;
         cp /workspace/run_benchmark.sh ./;
         cp /workspace/analysis_log.py ./;
         CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh fastscnn sp fp32 2 500 5;
         CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh fastscnn mp fp32 2 500 5;
         CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ocrnet_hrnetw48 sp fp32 2 500 5;
         CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh ocrnet_hrnetw48 mp fp32 2 500 5;
         CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh segformer_b0 sp fp32 2 500 5;
         CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh segformer_b0 mp fp32 2 500 5;
         "

nvidia-docker run --name test_torch_seg -it  \
    --net=host \
    --shm-size=1g \
    -v $PWD:/workspace \
    ${ImageName}  /bin/bash -c "${run_cmd}"

nvidia-docker stop test_torch_seg
nvidia-docker rm test_torch_seg