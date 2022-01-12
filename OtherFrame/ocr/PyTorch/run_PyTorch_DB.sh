#!/usr/bin/env bash

ImageName="registry.baidubce.com/paddlepaddle/paddle:2.0.2-gpu-cuda10.1-cudnn7"
#ImageName="paddlepaddle/paddle:latest-dev-cuda11.2-cudnn8-gcc82"
docker pull ${ImageName}

# 启动镜像后测试DB模型
run_cmd="cd /workspace;
         bash PrepareEnv.sh;
         cp /workspace/DB_scripts/* /workspace/models/DB/
         cd /workspace/models/DB/
         pip3.7 install -r requirement.txt
         bash prepare_data.sh
         bash run_benchmark.sh sp
         sleep 60
         bash run_benchmark.sh mp
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
