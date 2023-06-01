#!/usr/bin/env bash
#set -xe
echo "*******prepare benchmark***********"

################################# 创建一些log目录,如:
# export BENCHMARK_ROOT=/workspace 
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
pip install transformers==4.26.1 accelerate==0.16.0 pandas numpy scipy h5py tqdm 

python setup.py install

wget -nc -P ./data/ https://bj.bcebos.com/paddlenlp/datasets/benchmark_wikicorpus_en_seqlen128.tar --no-check-certificate
cd ./data/
tar -xf benchmark_wikicorpus_en_seqlen128.tar
cd ../

# 解决compile下报错的问题
ln -s /usr/include/python3.10 /usr/local/include/ 
echo "*******prepare benchmark end***********"
