#!/usr/bin/env bash
# 执行路径在模型库的根目录下
################################# 安装框架 如:
echo "*******prepare benchmark start ***********"
data_tiny_url="https://paddlespeech.bj.bcebos.com/datasets/dataset_tiny_AILITE"

if [ ${data_tiny_url} == "None" ]; then
	echo "Error!! please contact the author to get the URL"
	exit
fi

set -e

echo "https_proxy $HTTPS_PRO" 
echo "http_proxy $HTTP_PRO" 
export https_proxy=$HTTPS_PRO
export http_proxy=$HTTP_PRO
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com

# 修改竞品的训练日志
cp replace/executor.py wenet/utils/
cp replace/run.sh examples/aishell/s0
cp replace/download_and_untar.sh examples/aishell/s0/local
rm -f examples/aishell/s0/exp/conformer/*
rm -rf examples/librispeech/s1

WHEEL_URL_PREFIX="https://paddle-wheel.bj.bcebos.com/benchmark"
apt-get install -y axel
wget "$WHEEL_URL_PREFIX/torch-1.9.1%2Bcu111-cp37-cp37m-linux_x86_64.whl"
wget "$WHEEL_URL_PREFIX/torchvision-0.10.1%2Bcu111-cp37-cp37m-linux_x86_64.whl"
pip install torch-1.9.1+cu111-cp37-cp37m-linux_x86_64.whl
pip install torchvision-0.10.1+cu111-cp37-cp37m-linux_x86_64.whl
wget https://paddle-wheel.bj.bcebos.com/benchmark/torchaudio-0.9.1-cp37-cp37m-manylinux1_x86_64.whl
pip install torchaudio-0.9.1-cp37-cp37m-manylinux1_x86_64.whl
pip install -U importlib-metadata -i https://pypi.tuna.tsinghua.edu.cn/simple
apt-get install libsndfile1-dev -y
# pip install torch==1.9.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
#conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
#pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip list

# wget https://paddleseg.bj.bcebos.com/benchmark/mmseg/mmseg_benchmark_configs.tar.gz
# tar -zxf mmseg_benchmark_configs.tar.gz
################################# 准备训练数据 如:

cd examples/aishell/s0
mkdir -p data_store
export "device_gpu=1"
bash run.sh --stage -1 --stop_stage 3 --data_tiny_url ${data_tiny_url}
cd -

echo "*******prepare benchmark end***********"
