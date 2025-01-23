#!/usr/bin/env bash
#set -xe
echo $PWD

echo "*******prepare patch_code***********"
cp tokenizer_utils.py /opt/NeMo/nemo/collections/nlp/modules/common/tokenizer_utils.py 
cp exp_manager.py /opt/NeMo/nemo/utils/exp_manager.py

cp megatron_gpt_finetuning.py /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py
cp megatron_gpt_finetuning_config.yaml /opt/NeMo/examples/nlp/language_modeling/tuning/conf/megatron_gpt_finetuning_config.yaml

cp dpo.py /opt/NeMo-Aligner/nemo_aligner/algorithms/dpo.py
cp train_gpt_dpo.py /opt/NeMo-Aligner/examples/nlp/gpt/train_gpt_dpo.py
cp gpt_dpo.yaml /opt/NeMo-Aligner/examples/nlp/gpt/conf/gpt_dpo.yaml

echo "*******prepare benchmark***********"
export LD_LIBRARY_PATH=/home/opt/nvidia_lib:$LD_LIBRARY_PATH
python -m pip config set global.index-url https://pip.baidu-int.com/simple/
python -m pip config list

python -m pip install -U pip
python -m pip install nvitop

python -m pip install omegaconf
python -m pip install pytorch_lightning
python -m pip install hydra-core --upgrade

python -m pip install -U 'mlflow>=1.0.0'

export http_proxy=${HTTP_PRO}
export https_proxy=${HTTPS_PRO}
apt-get update -y
apt install iftop htop iotop -y --fix-missing
unset http_proxy && unset https_proxy

mkdir -p /opt/nemo-benchmark 
cp -r benchmark_yaml/* /opt/nemo-benchmark/

export no_proxy=bcebos.com
mkdir -p /opt/nemo-benchmark/models && cd /opt/nemo-benchmark/models
wget https://paddlenlp.bj.bcebos.com/llm_benchmark_data/NeMo_models_Qwen.tar.gz
tar zxf NeMo_models_Qwen.tar.gz && rm -rf NeMo_models_Qwen.tar.gz
cd -

cd /opt/nemo-benchmark
wget https://paddlenlp.bj.bcebos.com/llm_benchmark_data/NeMo_data-Qwen2.tar.gz
tar zxf NeMo_data-Qwen2.tar.gz && rm -rf NeMo_data-Qwen2.tar.gz
cd -

# mkdir -p /opt/models && cd /opt/models
# model_name_or_path=${model_name_or_path:-"Qwen2.5-1.5B"}
# axel -n 20 -q -c https://paddlenlp.bj.bcebos.com/models/huggingface/Qwen/${model_name_or_path}/${model_name_or_path}.nemo
# cd -
