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
  wget "$WHEEL_URL_PREFIX/torch-1.9.1%2Bcu111-cp37-cp37m-linux_x86_64.whl"
  wget "$WHEEL_URL_PREFIX/torchvision-0.10.1%2Bcu111-cp37-cp37m-linux_x86_64.whl"
  wget https://paddle-wheel.bj.bcebos.com/benchmark/mmcv_full-1.3.13-cp37-cp37m-linux_x86_64_A100.whl -O mmcv_full-1.3.13-cp37-cp37m-linux_x86_64.whl
  pip install torch-1.9.1+cu111-cp37-cp37m-linux_x86_64.whl
  pip install torchvision-0.10.1+cu111-cp37-cp37m-linux_x86_64.whl
else
  wget "$WHEEL_URL_PREFIX/torch-1.9.1-cp37-cp37m-manylinux1_x86_64.whl" 
  wget "$WHEEL_URL_PREFIX/torchvision-0.10.1-cp37-cp37m-manylinux1_x86_64.whl"
  wget https://paddle-wheel.bj.bcebos.com/benchmark/mmcv_full-1.3.13-cp37-cp37m-linux_x86_64_V100.whl -O mmcv_full-1.3.13-cp37-cp37m-linux_x86_64.whl
  pip install torch-1.9.1-cp37-cp37m-manylinux1_x86_64.whl
  pip install torchvision-0.10.1-cp37-cp37m-manylinux1_x86_64.whl
fi

# pip install torch==1.9.1 -i https://mirrors.ustc.edu.cn/pypi/web/simple
# pip install torchvision==0.10.1+cu111 -i https://pypi.tuna.tsinghua.edu.cn/simple
#pip install mmcv-full==1.3.13 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install mmcv_full-1.3.13-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e .
pip list

mkdir mmseg_benchmark_configs
mv deeplabv3p_resnet50.py mmseg_benchmark_configs/
cp dist_train.sh tools/dist_train.sh
################################# 准备训练数据 如:
mkdir -p data
echo "*******mkdir -p data***********"
wget https://paddleseg.bj.bcebos.com/tipc/data/cityscapes_300imgs.tar.gz \
    -O data/cityscapes_300imgs.tar.gz
tar -zxf data/cityscapes_300imgs.tar.gz -C data/
echo "*******prepare data finish***********"
rm -rf data/cityscapes
mv data/cityscapes_300imgs data/cityscapes
echo "*******prepare benchmark end***********"




