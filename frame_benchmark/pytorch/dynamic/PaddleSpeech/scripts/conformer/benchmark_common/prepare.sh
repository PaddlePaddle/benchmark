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

apt-get install -y axel
pip install -U importlib-metadata -i https://pypi.tuna.tsinghua.edu.cn/simple
apt-get install libsndfile1-dev -y
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
wget  ${FLAG_TORCH_WHL_URL}
tar xvf torch_dev_whls.tar
pip install torch_dev_whls/*
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
