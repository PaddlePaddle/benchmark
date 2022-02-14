#!/bin/bash
# 拷贝到 benmark 同级目录下执行
# 创建docker：registry.baidu.com/paddle-benchmark/paddlecloud-base-image:paddlecloud-ubuntu18.04-gcc8.2-cuda11.2-cudnn8
export ROOT_DIR=$PWD  # 设置个全局变量
cd ${ROOT_DIR}
shell_name=pwgan_bs6_fp32_SingleP_DP.sh
# shell_name=pwgan_bs26_fp32_SingleP_DP.sh
# shell_name=pwgan_bs6_fp32_MultiP_DP.sh
# shell_name=pwgan_bs26_fp32_MultiP_DP.sh
model_path=${ROOT_DIR}/benchmark/frame_benchmark/pytorch/dynamic/PaddleSpeech/models/ParallelWaveGAN
script_path=${ROOT_DIR}/benchmark/frame_benchmark/pytorch/dynamic/PaddleSpeech/scripts/pwgan/N1C1/${shell_name}
script_benchmark_common=${ROOT_DIR}/benchmark/frame_benchmark/pytorch/dynamic/PaddleSpeech/scripts/pwgan/benchmark_common/*
cp ${script_path} ${model_path}
cp ${script_benchmark_common} ${model_path} -r
cd ${model_path}
bash ${shell_name}
