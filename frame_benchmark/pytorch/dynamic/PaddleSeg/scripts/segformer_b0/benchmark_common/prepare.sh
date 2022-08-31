#!/usr/bin/env bash
# 执行路径在模型库的根目录下
################################# 安装框架 如:
echo "*******prepare benchmark start ***********"
#pip install -U pip
echo `pip --version`

CUDA_VERSION=`nvcc -run detect_cuda.cu | awk -F. '{print $1}'`

WHEEL_URL_PREFIX="https://paddle-wheel.bj.bcebos.com/benchmark"
apt-get install -y axel
if [ `nvidia-smi --list-gpus | grep A100 | wc -l` -ne "0" ]; then
  axel -n 16 "$WHEEL_URL_PREFIX/torch-1.9.1%2Bcu111-cp37-cp37m-linux_x86_64.whl"
  axel -n 16 "$WHEEL_URL_PREFIX/torchvision-0.10.1%2Bcu111-cp37-cp37m-linux_x86_64.whl"
  pip install torch-1.9.1+cu111-cp37-cp37m-linux_x86_64.whl
  pip install torchvision-0.10.1+cu111-cp37-cp37m-linux_x86_64.whl
else
  axel -n 16 "$WHEEL_URL_PREFIX/torch-1.9.1-cp37-cp37m-manylinux1_x86_64.whl" 
  axel -n 16 "$WHEEL_URL_PREFIX/torchvision-0.10.1-cp37-cp37m-manylinux1_x86_64.whl"
  pip install torch-1.9.1-cp37-cp37m-manylinux1_x86_64.whl
  pip install torchvision-0.10.1-cp37-cp37m-manylinux1_x86_64.whl
fi

#pip install torch==1.9.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
#pip install torchvision==0.10.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install mmcv-full==1.3.13 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e .
pip list

wget https://paddleseg.bj.bcebos.com/benchmark/mmseg/mmseg_benchmark_configs.tar.gz
tar -zxf mmseg_benchmark_configs.tar.gz
cp dist_train.sh tools/dist_train.sh
################################# 准备训练数据 如:
mkdir -p data
wget https://dataset.bj.bcebos.com/benchmark/cityscapes_300imgs.tar.gz \
    -O data/cityscapes_300imgs.tar.gz
tar -zxf data/cityscapes_300imgs.tar.gz -C data/
rm -rf data/cityscapes
mv data/cityscapes_300imgs data/cityscapes
echo "*******prepare benchmark end***********"




