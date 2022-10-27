#!/usr/bin/env bash
# 执行路径在模型库的根目录下
################################# 安装框架 如:
echo "*******prepare benchmark start ***********"
#pip install -U pip
echo `pip --version`

CUDA_VERSION=`nvcc -run detect_cuda.cu | awk -F. '{print $1}'`

WHEEL_URL_PREFIX="https://paddle-wheel.bj.bcebos.com/benchmark"
if [ `nvidia-smi --list-gpus | grep A100 | wc -l` -ne "0" ]; then
  wget "$WHEEL_URL_PREFIX/torch-1.9.1%2Bcu111-cp37-cp37m-linux_x86_64.whl"
  wget "$WHEEL_URL_PREFIX/torchvision-0.10.1%2Bcu111-cp37-cp37m-linux_x86_64.whl"
  pip install torch-1.9.1+cu111-cp37-cp37m-linux_x86_64.whl
  pip install torchvision-0.10.1+cu111-cp37-cp37m-linux_x86_64.whl
else
  wget "$WHEEL_URL_PREFIX/torch-1.9.1-cp37-cp37m-manylinux1_x86_64.whl" 
  wget "$WHEEL_URL_PREFIX/torchvision-0.10.1-cp37-cp37m-manylinux1_x86_64.whl"
  pip install torch-1.9.1-cp37-cp37m-manylinux1_x86_64.whl
  pip install torchvision-0.10.1-cp37-cp37m-manylinux1_x86_64.whl
fi

echo "PWD = $PWD"

# python3.7 ${BENCHMARK_ROOT}/paddlecloud/file_upload_download.py \
#     --remote-path frame_benchmark/pytorch_req/pytorch_191/ \
#     --local-path ./  \
#     --mode download
ls
# pip install torch==1.9.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
#pip install torch-1.9.1-cp37-cp37m-manylinux1_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
#pip install torchvision==0.10.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python
pip install -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple

wget -P /root/.cache/torch/hub/checkpoints/ https://paddle-wheel.bj.bcebos.com/benchmark/resnet50-0676ba61.pth

################################# 准备训练数据 如:
wget https://videotag.bj.bcebos.com/Data/k400_rawframes_small.tar \
    -O data/k400_rawframes_small.tar
tar -zxvf data/k400_rawframes_small.tar -C data/
echo "*******prepare benchmark end***********"