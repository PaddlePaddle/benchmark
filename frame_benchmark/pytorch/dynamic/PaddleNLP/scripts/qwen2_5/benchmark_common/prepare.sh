#!/usr/bin/env bash
#set -xe
echo "*******prepare benchmark***********"

################################# 创建一些log目录,如:
# export BENCHMARK_ROOT=/workspace   # 起容器的时候映射的目录  benchmark/OtherFrameworks/PyTorch/


export PATH=/opt/torch_native_venv/bin:${PATH}
export LD_LIBRARY_PATH=/home/opt/nvidia_lib:$LD_LIBRARY_PATH

echo $PWD
pip config set global.index-url http://pip.baidu.com/root/baidu/+simple/
pip config set install.trusted-host  pip.baidu.com
python -m pip install -U pip
python -m pip install torch==2.3.1 --extra-index https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
python -m pip install setuptools==61.0 --force-reinstall
python -m pip install -e .
python -m pip install deepspeed==0.14.2
python -m pip install modelscope

export http_proxy=${HTTP_PRO}
export https_proxy=${HTTPS_PRO}
model_name_or_path=${1:-"Qwen/Qwen2.5-1.5B"}
python -c "from modelscope import snapshot_download;model_dir = snapshot_download('${model_name_or_path}')"
unset http_proxy && unset https_proxy

mv -v data data_bak
wget https://paddlenlp.bj.bcebos.com/llm_benchmark_data/llamafactory_data.tar.gz
tar zxf llamafactory_data.tar.gz && rm -rf llamafactory_data.tar.gz
wget https://paddlenlp.bj.bcebos.com/llm_benchmark_data/deepspeed.tar.gz
tar zxf deepspeed.tar.gz && rm -rf deepspeed.tar.gz
