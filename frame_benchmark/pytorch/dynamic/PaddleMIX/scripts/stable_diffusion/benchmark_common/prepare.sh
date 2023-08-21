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
pip install install accelerate==0.20.3 transformers==4.30.2 pandas numpy scipy datasets diffusers==0.17.1 ftfy safetensors

rm -rf CompVis-stable-diffusion-v1-4-paddle-init-pt.tar.gz
wget https://bj.bcebos.com/paddlenlp/models/community/CompVis/CompVis-stable-diffusion-v1-4-paddle-init-pt.tar.gz
tar -zxvf CompVis-stable-diffusion-v1-4-paddle-init-pt.tar.gz

rm -rf laion400m_demo_data.tar.gz
rm -rf data
wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/laion400m_demo_data.tar.gz
tar -zxvf laion400m_demo_data.tar.gz

# 解决compile下报错的问题
ln -s /usr/include/python3.10 /usr/local/include/ 
echo "*******prepare benchmark end***********"
