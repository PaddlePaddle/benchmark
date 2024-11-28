#!/usr/bin/env bash
#set -xe
echo "*******prepare benchmark***********"

################################# 创建一些log目录,如:
# export BENCHMARK_ROOT=/workspace   # 起容器的时候映射的目录  benchmark/OtherFrameworks/PyTorch/
# github: https://github.com/pytorch/pytorch commit id: e525f433e15de1f16966901604a8c4c662828a8a


unset https_proxy && unset http_proxy

source /opt/torch_native_venv/bin/activate
cd LLaMA-Factory
python -m pip install -e .
pip install deepspeed
pip install modelscope
cd -

python -c "from modelscope import snapshot_download; \
    model_dir = snapshot_download('LLM-Research/Mistral-7B-Instruct-v0.3')"

wget https://paddlenlp.bj.bcebos.com/llm_benchmark_data/llamafactory/qwen2_5-7b_data.tar.gz
tar zxvf qwen2_5-7b_data.tar.gz