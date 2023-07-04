#!/usr/bin/env bash
#set -xe
echo "*******prepare benchmark***********"

################################# 创建一些log目录,如:
# export BENCHMARK_ROOT=/workspace   # 起容器的时候映射的目录  benchmark/OtherFrameworks/PyTorch/
# github: https://github.com/pytorch/pytorch commit id: e525f433e15de1f16966901604a8c4c662828a8a
run_env=$ROOT_DIR/run_env
log_date=`date "+%Y.%m%d.%H%M%S"`

unset https_proxy && unset http_proxy

wget -c --no-proxy ${FLAG_TORCH_WHL_URL}
tar_file_name=$(echo ${FLAG_TORCH_WHL_URL} | awk -F '/' '{print $NF}')
dir_name=$(echo ${tar_file_name} | awk -F '.tar' '{print $1}')
tar xf ${tar_file_name}
rm -rf ${tar_file_name}
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
pip install ${dir_name}/*
pip install install accelerate==0.19.0 transformers==4.29.2 networkx==2.6 sentencepiece numpy scipy datasets

rm -rf llama-7b-2l.tar
wget https://bj.bcebos.com/paddlenlp/models/community/facebook/llama-7b-2l.tar
tar -xvf llama-7b-2l.tar

rm -rf llama_sft_demo_data.tar.gz
rm -rf data
wget https://bj.bcebos.com/paddlenlp/models/community/facebook/llama_sft_demo_data.tar.gz
tar -xvf llama_sft_demo_data.tar.gz

# 解决compile下报错的问题
ln -s /usr/include/python3.10 /usr/local/include/ 
echo "*******prepare benchmark end***********"