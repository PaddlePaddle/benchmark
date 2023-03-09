#!/usr/bin/env bash
#set -xe
echo "*******prepare benchmark***********"

################################# 创建一些log目录,如:
# export BENCHMARK_ROOT=/workspace   # 起容器的时候映射的目录  benchmark/OtherFrameworks/PyTorch/
# github: https://github.com/pytorch/pytorch commit id: e525f433e15de1f16966901604a8c4c662828a8a
run_env=$ROOT_DIR/run_env
log_date=`date "+%Y.%m%d.%H%M%S"`

unset https_proxy && unset http_proxy

wget ${FLAG_TORCH_WHL_URL}
tar -xvf torch_dev_whls.tar
pip install torch_dev_whls/*
pip install transformers==4.26.1 accelerate==0.16.0 pandas numpy scipy datasets diffusers==0.11.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

python setup.py install

wget https://bj.bcebos.com/paddlenlp/models/community/CompVis/CompVis-stable-diffusion-v1-4-pt.tar.gz
tar -zxvf CompVis-stable-diffusion-v1-4-pt.tar.gz

wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/pokemon-blip-captions.tar.gz
tar -zxvf pokemon-blip-captions.tar.gz

echo "*******prepare benchmark end***********"
