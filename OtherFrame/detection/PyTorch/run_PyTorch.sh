#!/usr/bin/env bash

ImageName="registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7";
docker pull ${ImageName}
export BENCHMARK_ROOT=/workspace # 对应实际地址 benchmark/OtherFrameworks/video/PyTorch

run_cmd="cd ${BENCHMARK_ROOT}
        bash PrepareEnv.sh
        cd ${BENCHMARK_ROOT}/models/mmdetection;
        cp ${BENCHMARK_ROOT}/scripts/run_benchmark.sh ./;
        cp ${BENCHMARK_ROOT}/scripts/analysis_log.py ./;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 1 fp32 1 faster_rcnn;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 1 fp32 1 faster_rcnn;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 8 fp32 1 faster_rcnn;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 8 fp32 1 faster_rcnn;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 2 fp32 1 fcos;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 2 fp32 1 fcos;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 8 fp32 1 fcos;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 8 fp32 1 fcos;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 2 fp32 1 deformable_detr;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 2 fp32 1 deformable_detr;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 2 fp32 1 gfl;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 2 fp32 1 gfl;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 8 fp32 1 gfl;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 8 fp32 1 gfl;
        sleep 60;

        cd ${BENCHMARK_ROOT}/models/mmpose;
        cp ${BENCHMARK_ROOT}/scripts/run_benchmark.sh ./;
        cp ${BENCHMARK_ROOT}/scripts/analysis_log.py ./;
        pip uninstall -y mmcv mmcv-full
        pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.1/index.html
        pip install -r requirements.txt
        pip install -v -e .
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 64 fp32 10 hrnet_w32_keypoint;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 64 fp32 10 hrnet_w32_keypoint;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 160 fp32 10 hrnet_w32_keypoint;
        sleep 60;
        cuda_visible_devices=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 160 fp32 10 hrnet_w32_keypoint;
        sleep 60;

        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 20 fp32 1 higherhrnet_w32;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 20 fp32 1 higherhrnet_w32;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 24 fp32 1 higherhrnet_w32;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 24 fp32 1 higherhrnet_w32;
        sleep 60;


        cd ${BENCHMARK_ROOT}/models/SOLO;
        pip uninstall -y mmdet mmcv mmcv-full
        pip install mmcv
        pip install -r requirements.txt
        pip install -v -e .
        cp ${BENCHMARK_ROOT}/scripts/run_benchmark.sh ./;
        cp ${BENCHMARK_ROOT}/scripts/analysis_log.py ./;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 2 fp32 1 solov2;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 2 fp32 1 solov2;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 4 fp32 1 solov2;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 4 fp32 1 solov2;
        sleep 60;
        "

nvidia-docker run --name test_torch_detection -it  \
    --net=host \
    --shm-size=64g \
    -v $PWD:/workspace \
    ${ImageName}  /bin/bash -c "${run_cmd}"

nvidia-docker stop test_torch_detection
nvidia-docker rm test_torch_detection
