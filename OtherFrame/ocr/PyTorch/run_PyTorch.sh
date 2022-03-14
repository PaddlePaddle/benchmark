#!/usr/bin/env bash

ImageName="paddlepaddle/paddle:latest-dev-cuda11.2-cudnn8-gcc82"
docker pull ${ImageName}

# 启动镜像后测试PSE模型
nvidia-docker stop test_torch_ocr
nvidia-docker rm test_torch_ocr 
run_cmd="cd /workspace;
         \cp -f /workspace/PrepareEnv.sh ./;
         bash PrepareEnv.sh;

         cd /workspace/models/PSENet/;
         pip3.7 install -r requirement.txt;
         \cp -f /workspace/scripts/PSENet/preData.sh ./;
         \cp -f /workspace/scripts/PSENet/run_benchmark.sh ./;
         \cp -f /workspace/scripts/PSENet/analysis_log.py ./;
         bash preData.sh;
         sed -i '/set\ -xe/d' run_benchmark.sh
         CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 8 fp32 2 > ocr_pse_bs8_fp32_sp.speed 2>&1;
         sleep 60
         CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 64 fp32 2 > ocr_pse_bs8_fp32_mp.speed 2>&1;
         sleep 60
         CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 16 fp32 2 > ocr_pse_bs16_fp32_sp.speed 2>&1;
         sleep 60
         CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 128 fp32 2 > ocr_pse_bs16_fp32_mp.speed 2>&1;
        "

nvidia-docker run --name test_torch_ocr -i \
    --net=host \
    --shm-size=128g \
    -v $PWD:/workspace \
      -v /ssd3:/ssd3 \
    -v /ssd2:/ssd2 \
    -e "ALL_PATH=${all_path}" \
    -v "BENCHMARK_ROOT=/workspace" \
    -e "http_proxy=${http_proxy}" \
    -e "https_proxy=${http_proxy}" \
    -e "no_proxy=bcebos.com" \
    ${ImageName} /bin/bash -c "${run_cmd}"

nvidia-docker stop test_torch_ocr
nvidia-docker rm test_torch_ocr 
