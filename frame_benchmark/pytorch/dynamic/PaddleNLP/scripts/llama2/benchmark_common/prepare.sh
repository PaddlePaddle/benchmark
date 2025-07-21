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
python -m pip install liger-kernel==0.4.2

model_name_or_path=${1:-"meta-llama/Llama-2-7b-hf"}
mkdir -p /opt/${model_name_or_path} && cd /opt/${model_name_or_path}
export no_proxy=bcebos.com
case ${model_name_or_path} in
shakechen/Llama-2-7b-hf)
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/shakechen/Llama-2-7b-hf/LICENSE.txt
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/shakechen/Llama-2-7b-hf/README.md
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/shakechen/Llama-2-7b-hf/Responsible-Use-Guide.pdf
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/shakechen/Llama-2-7b-hf/USE_POLICY.md
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/shakechen/Llama-2-7b-hf/config.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/shakechen/Llama-2-7b-hf/configuration.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/shakechen/Llama-2-7b-hf/generation_config.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/shakechen/Llama-2-7b-hf/model-00001-of-00002.safetensors
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/shakechen/Llama-2-7b-hf/model-00002-of-00002.safetensors
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/shakechen/Llama-2-7b-hf/model.safetensors.index.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/shakechen/Llama-2-7b-hf/pytorch_model-00001-of-00002.bin
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/shakechen/Llama-2-7b-hf/pytorch_model-00002-of-00002.bin
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/shakechen/Llama-2-7b-hf/pytorch_model.bin.index.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/shakechen/Llama-2-7b-hf/special_tokens_map.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/shakechen/Llama-2-7b-hf/tokenizer.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/shakechen/Llama-2-7b-hf/tokenizer.model
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/shakechen/Llama-2-7b-hf/tokenizer_config.json
    echo "download models for shakechen/Llama-2-7b-hf done" ;;
ydyajyA/Llama-2-13b-chat-hf)
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/ydyajyA/Llama-2-13b-chat-hf/config.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/ydyajyA/Llama-2-13b-chat-hf/configuration.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/ydyajyA/Llama-2-13b-chat-hf/generation_config.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/ydyajyA/Llama-2-13b-chat-hf/model-00001-of-00006.safetensors
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/ydyajyA/Llama-2-13b-chat-hf/model-00002-of-00006.safetensors
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/ydyajyA/Llama-2-13b-chat-hf/model-00003-of-00006.safetensors
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/ydyajyA/Llama-2-13b-chat-hf/model-00004-of-00006.safetensors
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/ydyajyA/Llama-2-13b-chat-hf/model-00005-of-00006.safetensors
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/ydyajyA/Llama-2-13b-chat-hf/model-00006-of-00006.safetensors
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/ydyajyA/Llama-2-13b-chat-hf/model.safetensors.index.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/ydyajyA/Llama-2-13b-chat-hf/pytorch_model.bin.index.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/ydyajyA/Llama-2-13b-chat-hf/special_tokens_map.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/ydyajyA/Llama-2-13b-chat-hf/tokenizer.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/ydyajyA/Llama-2-13b-chat-hf/tokenizer.model
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/ydyajyA/Llama-2-13b-chat-hf/tokenizer_config.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/ydyajyA/Llama-2-13b-chat-hf/up.ipynb
    echo "download models for ydyajyA/Llama-2-13b-chat-hf done" ;;
meta-llama/Llama-2-70b-hf)
    nums1=("00001" "00002" "00003" "00004" "00005")
    for num in "${nums1[@]}"; do
        url="https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-${num}-of-00015.safetensors"
        wget -c "${url}" &
    done
    wait
    nums2=("00006" "00007" "00008" "00009" "00010")
    for num in "${nums2[@]}"; do
        url="https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-${num}-of-00015.safetensors"
        wget -c "${url}" &
    done
    wait
    nums3=("00011" "00012" "00013" "00014" "00015")
    for num in "${nums3[@]}"; do
        url="https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-${num}-of-00015.safetensors"
        wget -c "${url}" &
    done
    wait

    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/LICENSE.txt
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/README.md
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/Responsible-Use-Guide.pdf
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/USE_POLICY.md
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/config.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/configuration.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/generation_config.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/llama_updates.patch
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00001-of-00015.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00002-of-00015.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00003-of-00015.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00004-of-00015.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00005-of-00015.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00006-of-00015.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00007-of-00015.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00008-of-00015.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00009-of-00015.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00010-of-00015.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00011-of-00015.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00012-of-00015.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00013-of-00015.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00014-of-00015.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model-00015-of-00015.safetensors
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/model.safetensors.index.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/special_tokens_map.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/tokenizer.model
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/meta-llama/Llama-2-70b-hf/tokenizer_config.json
    echo "download models for meta-llama/Llama-2-70b-hf done" ;;
*) 
    echo "${model_name_or_path} not in bos, download from modelscope"; 
    python -c "from modelscope import snapshot_download;\
        model_dir = snapshot_download('${model_name_or_path}', ignore_file_pattern=[\"*.safetensors\", \"*.bin\"])"
    ln -s /root/.cache/modelscope/hub/${model_name_or_path} /opt/${model_name_or_path}
    echo "download models for ${model_name_or_path} done" ;;
esac
cd -


mv -v data data_bak
wget --no-proxy -c https://paddlenlp.bj.bcebos.com/llm_benchmark_data/llamafactory_data.tar.gz
tar zxf llamafactory_data.tar.gz && rm -rf llamafactory_data.tar.gz
wget --no-proxy -c https://paddlenlp.bj.bcebos.com/llm_benchmark_data/deepspeed.tar.gz
tar zxf deepspeed.tar.gz && rm -rf deepspeed.tar.gz
