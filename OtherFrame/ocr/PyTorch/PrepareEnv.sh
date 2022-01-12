#!/usr/bin/env bash

# 公共配置文件,配置python 安装pytorch,运行目录:/workspace (起容器的时候映射的目录:benchmark/OtherFrameworks/ocr/PyTorch)
echo "*******prepare benchmark***********"

################################# 创建一些log目录,如:
export BENCHMARK_ROOT=/workspace
# this for update the log_path coding mat

log_path=${TRAIN_LOG_DIR:-"pwd"}

################################# 安装框架 如:
pip3.7 install -U pip -i https://mirror.baidu.com/pypi/simple
echo `pip --version`
pip3.7 install pqi -i https://mirror.baidu.com/pypi/simple
pqi add baidu https://mirror.baidu.com/pypi/simple
pqi use baidu

#pip3.7 install torch==1.10.0+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip3.7 install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip3.7 install anyconfig
echo "*******prepare benchmark end***********"

