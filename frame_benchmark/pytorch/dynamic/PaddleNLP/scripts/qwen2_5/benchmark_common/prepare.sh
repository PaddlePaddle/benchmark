#!/usr/bin/env bash
#set -xe
echo "*******prepare benchmark***********"

################################# 创建一些log目录,如:
# export BENCHMARK_ROOT=/workspace   # 起容器的时候映射的目录  benchmark/OtherFrameworks/PyTorch/


export http_proxy=${HTTP_PRO}
export https_proxy=${HTTPS_PRO}
source /opt/torch_native_venv/bin/activate
export LD_LIBRARY_PATH=/home/opt/nvidia_lib:$LD_LIBRARY_PATH

echo $PWD
python -m pip install -U pip
python -m pip install setuptools==61.0 --force-reinstall
python -m pip install -e .
python -m pip install deepspeed==0.14.2
python -m pip install modelscope

mv -v data data_bak
mv -v data data_bak
wget https://paddlenlp.bj.bcebos.com/llm_benchmark_data/llamafactory_data.tar.gz
tar zxvf llamafactory_data.tar.gz && rm -rf llamafactory_data.tar.gz
wget https://paddlenlp.bj.bcebos.com/llm_benchmark_data/deepspeed.tar.gz
tar zxvf deepspeed.tar.gz && rm -rf deepspeed.tar.gz
