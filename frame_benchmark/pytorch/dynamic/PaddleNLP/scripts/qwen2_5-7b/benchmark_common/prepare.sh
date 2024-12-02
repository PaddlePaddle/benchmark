#!/usr/bin/env bash
#set -xe
echo "*******prepare benchmark***********"

################################# 创建一些log目录,如:
# export BENCHMARK_ROOT=/workspace   # 起容器的时候映射的目录  benchmark/OtherFrameworks/PyTorch/


unset https_proxy && unset http_proxy

source /opt/torch_native_venv/bin/activate
echo $pwd
python -m pip install -e .
pip install deepspeed
pip install modelscope

python -c "from modelscope import snapshot_download \
            model_dir = snapshot_download('Qwen/Qwen2.5-7B'))"
python -c "from modelscope import snapshot_download \
            model_dir = snapshot_download('Qwen/Qwen2.5-14B'))"

wget https://paddlenlp.bj.bcebos.com/llm_benchmark_data/qwen2_5_llamafactory_data.tar.gz
tar zxvf qwen2_5_llamafactory_data.tar.gz && rm -rf qwen2_5_llamafactory_data.tar.gz