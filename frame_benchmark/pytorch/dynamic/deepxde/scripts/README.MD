# DeepXDE 下动态图benchmark模型执行说明
动态图benchmark测试脚本说明
# 目录说明 
# Docker 运行环境
docker image: registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.2-cudnn8-gcc82
paddle = 0.0.0
python = 3.7
# 运行benchmark测试步骤
git clone https://github.com/PaddleBenchmark/deepxde -b pytorch_benchmark
cd deepxde
# 运行模型
## 单卡（自动运行打开Profiling）
export CUDA_VISIBLE_DEVICES=0 
bash scripts/Eular_beam/N1C1/Eular_beam_bs1_fp32_DP.sh 
bash scripts/Navier_Stokes_inverse/N1C1/Navier_Stokes_inverse_bs1_fp32_DP.sh 
## 多卡(无）
# profiling（无）
