#!/usr/bin/env bash

ImageName="registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7";
docker pull ${ImageName}

run_cmd="cp /workspace/scripts/PrepareEnv.sh ./;
         bash PrepareEnv.sh;
         cd /workspace/models/mmedi;
         cp -r /workspace/mmedi_benchmark_configs ./;
         cp /workspace/scripts/run_benchmark.sh ./;
         cp /workspace/scripts/analysis_log.py ./;
         sed -i '/set\ -xe/d' run_benchmark.sh
         PORT=23335 CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh esrgan_bs32_fp32 sp fp32 32 300 4;
         PORT=23335 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh esrgan_bs32_fp32 mp fp32 32 300 4;
         PORT=23335 CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh edvr_bs4_fp32 sp fp32 4 300 3;
         PORT=23335 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh edvr_bs4_fp32 mp fp32 4 300 3;
         PORT=23335 CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh basicvsr_bs2_fp32 sp fp32 2 300 4;
         PORT=23335 CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh basicvsr_bs4_fp32 sp fp32 4 300 4;
         PORT=23335 CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_benchmark.sh basicvsr_bs2_fp32 mp fp32 2 300 4;
         PORT=23335 CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_benchmark.sh basicvsr_bs4_fp32 mp fp32 4 300 4;
         "
         #PORT=23335 CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh esrgan_bs64_fp32 sp fp32 64 300 4;
         #PORT=23335 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh esrgan_bs64_fp32 mp fp32 64 300 4;
         #PORT=23335 CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh edvr_bs64_fp32 sp fp32 64 300 3;
         #PORT=23335 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh edvr_bs64_fp32 mp fp32 64 300 3;

nvidia-docker run --name test_torch_gan -i  \
    --net=host \
    --shm-size=128g \
    -v $PWD:/workspace \
    -v /ssd2:/ssd2 \
    -e "ALL_PATH=${all_path}" \
    -v "BENCHMARK_ROOT=/workspace" \
    -e "http_proxy=${http_proxy}" \
    -e "https_proxy=${http_proxy}" \
    -e "no_proxy=bcebos.com" \
    ${ImageName}  /bin/bash -c "${run_cmd}"

nvidia-docker stop test_torch_gan
nvidia-docker rm test_torch_gan
