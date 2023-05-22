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
pip install install accelerate==0.19.0 transformers==4.29.1 pandas numpy scipy datasets diffusers==0.16.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

python setup.py install

rm -rf CompVis-ldm-text2im-large-256-pt.tar.gz
wget https://bj.bcebos.com/paddlenlp/models/community/CompVis/CompVis-ldm-text2im-large-256-pt.tar.gz
tar -zxvf CompVis-ldm-text2im-large-256-pt.tar.gz

rm -rf laion400m_demo_data.tar.gz
rm -rf data
wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/laion400m_demo_data.tar.gz
tar -zxvf laion400m_demo_data.tar.gz

echo "*******prepare benchmark end***********"
