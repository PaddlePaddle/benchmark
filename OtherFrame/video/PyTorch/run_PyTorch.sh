# 提交内容 #
ImageName="registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7";
docker pull ${ImageName}
export BENCHMARK_ROOT=/workspace # 对应实际地址 benchmark/OtherFrameworks/video/PyTorch

run_cmd="bash PrepareEnv.sh
        cd ${BENCHMARK_ROOT}/models/TimeSformer;
        cp ${BENCHMARK_ROOT}/scripts/TimeSformer/run_benchmark.sh ./;
        cp ${BENCHMARK_ROOT}/scripts/TimeSformer/analysis_log.py ./;
        cp ${BENCHMARK_ROOT}/scripts/TimeSformer/preData.sh ./;
        bash preData.sh;

        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 1 fp32 TimeSformer;
        sleep 60;

        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 1 fp32 TimeSformer;
        sleep 60;

        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 14 fp32 TimeSformer;
        sleep 60;

        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 14 fp32 TimeSformer;
        sleep 60;
        "

nvidia-docker run --name test_torch_video -it  \
    --net=host \
    --shm-size=64g \
    -v $PWD:/workspace \
    ${ImageName}  /bin/bash -c "${run_cmd}"

nvidia-docker stop test_torch_video
nvidia-docker rm test_torch_video
