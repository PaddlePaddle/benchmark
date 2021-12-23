#!/usr/bin/env bash

ImageName="registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7";
docker pull ${ImageName}

run_cmd="cp /workspace/scripts/PrepareEnv.sh ./;
         bash PrepareEnv.sh;
         cd /workspace/models/mmseg;
         cp /workspace/scripts/run_benchmark.sh ./;
         cp /workspace/scripts/analysis_log.py ./;
         CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh fastscnn sp fp32 2 500 5;
         CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh fastscnn mp fp32 2 500 5;
         CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ocrnet_hrnetw48 sp fp32 2 500 5;
         CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh ocrnet_hrnetw48 mp fp32 2 500 5;
         CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh segformer_b0 sp fp32 2 500 5;
         CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh segformer_b0 mp fp32 2 500 5;
         "
         # 暂时不跑
         #CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh fastscnn sp fp32 4 500 5;
         #CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh fastscnn mp fp32 4 500 5;
         #CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh segformer_b0 sp fp32 4 500 5;
         #CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh segformer_b0 mp fp32 4 500 5;
         #CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ocrnet_hrnetw48 sp fp32 4 500 5;
         #CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh ocrnet_hrnetw48 mp fp32 4 500 5;

nvidia-docker run --name test_torch_seg -i  \
    --net=host \
    --shm-size=1g \
    -e "http_proxy=${http_proxy}" \
    -e "https_proxy=${https_proxy}" \
    -e "no_proxy=bcebos.com" \
    -v $PWD:/workspace \
    ${ImageName}  /bin/bash -c "${run_cmd}"

nvidia-docker stop test_torch_seg
nvidia-docker rm test_torch_seg
