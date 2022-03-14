#!/usr/bin/env bash
# 执行路径在竞品模型库的根目录下

stage=0
stop_stage=100

echo "*******prepare benchmark start ***********"

set -e 

################################# 安装框架 如:

# 安装 pyrtorch
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    pip install pytest-runner -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install -U pip
    echo `pip --version`
    echo "install pytorch..."
    pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    echo "pytorch installed"
fi
# 拉取模型代码并安装
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "install pwgan..."
    pip install .
    echo "pwgan installed"
    echo "install apex..."
    # If you want to use distributed training, please run following command to install apex.
    git clone https://github.com/NVIDIA/apex.git
    pushd apex
    git checkout 3303b3e
    pip install -v --disable-pip-version-check --no-cache-dir ./
    pushd ../
    echo "apex installed"

    # install nkf
    apt-get install nkf -y
    apt-get install sox -y
    apt-get install jq -y

fi

################################# 准备训练数据 如:
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    pushd egs/csmsc/voc1
    bash run.sh --stage -1 --stop-stage 1
    popd
fi
echo "*******prepare benchmark end***********"
