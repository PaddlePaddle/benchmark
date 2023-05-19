#!/usr/bin/env bash
#set -xe
echo "*******prepare benchmark***********"

################################# 创建一些log目录,如:
# export BENCHMARK_ROOT=/workspace 
run_env=$ROOT_DIR/run_env
log_date=`date "+%Y.%m%d.%H%M%S"`

unset https_proxy && unset http_proxy

wget ${FLAG_TORCH_WHL_URL}
tar -xvf torch_dev_whls.tar
pip install torch_dev_whls/*
pip install transformers==4.26.1 accelerate==0.16.0 pandas numpy scipy h5py tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple

python setup.py install

wget -nc -P ./data/ https://bj.bcebos.com/paddlenlp/datasets/benchmark_wikicorpus_en_seqlen128.tar --no-check-certificate
cd ./data/
tar -xf benchmark_wikicorpus_en_seqlen128.tar
cd ../

echo "*******prepare benchmark end***********"
