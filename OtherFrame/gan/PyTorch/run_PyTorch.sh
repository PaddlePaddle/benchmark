#!/usr/bin/env bash

ImageName="registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7";
docker pull ${ImageName}

run_cmd="cp /workspace/scripts/PrepareEnv.sh ./;
         bash PrepareEnv.sh;
         cd /workspace/models/mmedi;
         cp /workspace/scripts/run_benchmark.sh ./;
         cp /workspace/scripts/analysis_log.py ./;
         CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh esrgan sp fp32 32 300 4;
         CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh esrgan sp fp32 64 300 4;
         CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh esrgan mp fp32 32 300 4;
         CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh esrgan mp fp32 64 300 4;
         CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh edvr sp fp32 4 300 3;
         CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh edvr sp fp32 64 300 3;
         CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh edvr mp fp32 4 300 3;
         CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh edvr mp fp32 64 300 3;
         CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh basicvsr sp fp32 2 300 4;
         CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh basicvsr sp fp32 4 300 4;
         CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_benchmark.sh basicvsr mp fp32 2 300 4;
         CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_benchmark.sh basicvsr mp fp32 4 300 4;
         "

nvidia-docker run --name test_torch_gan -it  \
    --net=host \
    --shm-size=128g \
    -v $PWD:/workspace \
    ${ImageName}  /bin/bash -c "${run_cmd}"

nvidia-docker stop test_torch_gan
nvidia-docker rm test_torch_gan