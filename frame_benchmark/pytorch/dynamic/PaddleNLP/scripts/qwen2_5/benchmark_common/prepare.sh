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

model_name_or_path=${1:-"Qwen/Qwen2.5-1.5B"}
mkdir -p /opt/${model_name_or_path} && cd /opt/${model_name_or_path}
export no_proxy=bcebos.com
case ${model_name_or_path} in
Qwen/Qwen2.5-1.5B)  
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-1.5B/LICENSE
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-1.5B/README.md
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-1.5B/config.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-1.5B/configuration.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-1.5B/generation_config.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-1.5B/merges.txt
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-1.5B/model.safetensors
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-1.5B/tokenizer.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-1.5B/tokenizer_config.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-1.5B/vocab.json
    echo "download models for Qwen/Qwen2.5-1.5B done" ;;
Qwen/Qwen2.5-7B) 
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-7B/LICENSE
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-7B/README.md
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-7B/config.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-7B/configuration.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-7B/generation_config.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-7B/merges.txt
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-7B/model-00001-of-00004.safetensors
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-7B/model-00002-of-00004.safetensors
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-7B/model-00003-of-00004.safetensors
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-7B/model-00004-of-00004.safetensors
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-7B/model.safetensors.index.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-7B/tokenizer.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-7B/tokenizer_config.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-7B/vocab.json
    echo "download models for Qwen/Qwen2.5-7B done" ;;
Qwen/Qwen2.5-14B)  
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-14B/LICENSE
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-14B/README.md
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-14B/config.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-14B/configuration.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-14B/generation_config.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-14B/index.html
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-14B/merges.txt
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-14B/model-00001-of-00006.safetensors
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-14B/model-00001-of-00008.safetensors
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-14B/model-00002-of-00006.safetensors
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-14B/model-00002-of-00008.safetensors
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-14B/model-00003-of-00006.safetensors
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-14B/model-00003-of-00008.safetensors
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-14B/model-00004-of-00006.safetensors
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-14B/model-00004-of-00008.safetensors
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-14B/model-00005-of-00006.safetensors
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-14B/model-00005-of-00008.safetensors
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-14B/model-00006-of-00006.safetensors
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-14B/model-00006-of-00008.safetensors
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-14B/model-00007-of-00008.safetensors
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-14B/model-00008-of-00008.safetensors
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-14B/model.safetensors.index.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-14B/tokenizer.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-14B/tokenizer_config.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-14B/vocab.json
    echo "download models for Qwen/Qwen2.5-14B done" ;;
Qwen/Qwen2.5-72B)  
    nums1=("00001" "00002" "00003" "00004" "00005" "00006" "00007")
    for num in "${nums1[@]}"; do
        url="https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-${num}-of-00037.safetensors"
        wget -c "${url}" &
    done
    wait
    nums2=("00008" "00009" "00010" "00011" "00012" "00013")
    for num in "${nums2[@]}"; do
        url="https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-${num}-of-00037.safetensors"
        wget -c "${url}" &
    done
    wait
    nums3=("00014" "00015" "00016" "00017" "00018" "00019")
    for num in "${nums3[@]}"; do
        url="https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-${num}-of-00037.safetensors"
        wget -c "${url}" &
    done
    wait
    nums4=("00020" "00021" "00022" "00023" "00024" "00025")
    for num in "${nums4[@]}"; do
        url="https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-${num}-of-00037.safetensors"
        wget -c "${url}" &
    done
    wait
    nums5=("00026" "00027" "00028" "00029" "00030" "00031")
    for num in "${nums5[@]}"; do
        url="https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-${num}-of-00037.safetensors"
        wget -c "${url}" &
    done
    wait
    nums6=("00032" "00033" "00034" "00035" "00036" "00037")
    for num in "${nums6[@]}"; do
        url="https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-${num}-of-00037.safetensors"
        wget -c "${url}" &
    done
    wait

    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/LICENSE
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/README.md
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/config.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/configuration.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/generation_config.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/merges.txt
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00001-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00002-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00003-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00004-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00005-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00006-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00007-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00008-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00009-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00010-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00011-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00012-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00013-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00014-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00015-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00016-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00017-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00018-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00019-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00020-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00021-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00022-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00023-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00024-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00025-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00026-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00027-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00028-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00029-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00030-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00031-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00032-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00033-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00034-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00035-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00036-of-00037.safetensors
    # wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model-00037-of-00037.safetensors
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/model.safetensors.index.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/tokenizer.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/tokenizer_config.json
    wget --no-proxy -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/Qwen2.5-72B/vocab.json
    echo "download models for Qwen/Qwen2.5-72B done" ;;
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
