#!/usr/bin/env bash
#set -xe
echo "*******prepare benchmark***********"

################################# 创建一些log目录,如:
# export BENCHMARK_ROOT=/workspace   # 起容器的时候映射的目录  benchmark/OtherFrameworks/PyTorch/


export PATH=/opt/torch_native_venv/bin:${PATH}
export LD_LIBRARY_PATH=/home/opt/nvidia_lib:$LD_LIBRARY_PATH

echo $PWD
python -m pip config set global.index-url https://pip.baidu-int.com/simple/
python -m pip config list
python -m pip install -U pip
python -m pip install setuptools==61.0 --force-reinstall
python -m pip install torch==2.3.1
python -m pip install -e .
python -m pip install deepspeed==0.14.2
python -m pip install modelscope

model_name_or_path=${1:-"meta-llama/Llama-2-7b-hf"}
mkdir -p /opt/${model_name_or_path} && cd /opt/${model_name_or_path}
case ${model_name_or_path} in
meta-llama/Llama-2-7b-hf)
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-7b-hf/README.md
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-7b-hf/USE_POLICY.md
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-7b-hf/config.json
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-7b-hf/generation_config.json
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-7b-hf/model-00001-of-00002.safetensors
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-7b-hf/model-00002-of-00002.safetensors
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-7b-hf/model.safetensors.index.json
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-7b-hf/special_tokens_map.json
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-7b-hf/tokenizer.json
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-7b-hf/tokenizer.model
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-7b-hf/tokenizer_config.json
    echo "download models for meta-llama/Llama-2-7b-hf done" ;;
meta-llama/Llama-2-13b-hf)
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-13b-hf/LICENSE.txt
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-13b-hf/README.md
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-13b-hf/Responsible-Use-Guide.pdf
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-13b-hf/USE_POLICY.md
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-13b-hf/config.json
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-13b-hf/configuration.json
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-13b-hf/generation_config.json
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-13b-hf/model-00001-of-00003.safetensors
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-13b-hf/model-00002-of-00003.safetensors
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-13b-hf/model-00003-of-00003.safetensors
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-13b-hf/model.safetensors.index.json
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-13b-hf/special_tokens_map.json
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-13b-hf/tokenizer.model
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-13b-hf/tokenizer_config.json
    echo "download models for meta-llama/Llama-2-13b-hf done" ;;
meta-llama/Llama-2-70b-hf)
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/LICENSE.txt
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/README.md
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/Responsible-Use-Guide.pdf
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/USE_POLICY.md
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/config.json
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/configuration.json
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/generation_config.json
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/llama_updates.patch
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00001-of-00015.safetensors
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00002-of-00015.safetensors
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00003-of-00015.safetensors
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00004-of-00015.safetensors
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00005-of-00015.safetensors
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00006-of-00015.safetensors
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00007-of-00015.safetensors
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00008-of-00015.safetensors
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00009-of-00015.safetensors
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00010-of-00015.safetensors
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00011-of-00015.safetensors
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00012-of-00015.safetensors
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00013-of-00015.safetensors
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00014-of-00015.safetensors
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00015-of-00015.safetensors
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model.safetensors.index.json
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/special_tokens_map.json
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/tokenizer.model
    wget https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/tokenizer_config.json
    echo "download models for meta-llama/Llama-2-70b-hf done" ;;
*) echo "no ${model_name_or_path}"; exit 1;
esac


mv -v data data_bak
wget https://paddlenlp.bj.bcebos.com/llm_benchmark_data/llamafactory_data.tar.gz
tar zxf llamafactory_data.tar.gz && rm -rf llamafactory_data.tar.gz
wget https://paddlenlp.bj.bcebos.com/llm_benchmark_data/deepspeed.tar.gz
tar zxf deepspeed.tar.gz && rm -rf deepspeed.tar.gz
