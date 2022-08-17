#!/usr/bin/env bash
# 执行路径在竞品模型库的根目录下

stage=0
stop_stage=100

echo "*******prepare benchmark start ***********"

set -e 

################################# 安装框架 如:

rm -rf apex/
WHEEL_URL_PREFIX="https://paddle-wheel.bj.bcebos.com/benchmark"
pip install setuptools==50.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
# 安装 pyrtorch
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    pip install pytest-runner -i https://pypi.tuna.tsinghua.edu.cn/simple
    #pip install -U pip
    echo `pip --version`
    echo "install pytorch..."
    wget "$WHEEL_URL_PREFIX/torch-1.9.1%2Bcu111-cp37-cp37m-linux_x86_64.whl"
    pip install torch-1.9.1+cu111-cp37-cp37m-linux_x86_64.whl
    #pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    echo "pytorch installed"
fi
# 拉取模型代码并安装
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    apt-get install jq
    pip install librosa==0.8.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install kaldiio -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install h5py -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo "install pwgan..."
    pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo "pwgan installed"
    echo "install apex..."
    # If you want to use distributed training, please run following command to install apex.
    git clone http://github.com/NVIDIA/apex.git
    pushd apex
    git checkout 3303b3e
    pip install -v --disable-pip-version-check --no-cache-dir ./
    pushd ../
    echo "apex installed"

    # install nkf
    apt-get install nkf -y
    echo "apt-get install nkf -y"
    apt-get install sox -y
    echo "apt-get install sox -y"
    apt-get install jq -y
    echo " apt-get install jq -y"

fi

################################# 准备训练数据 如:
cp data_download.sh egs/csmsc/voc1/local/data_download.sh
cp data_prep.sh egs/csmsc/voc1/local/data_prep.sh
cp run.sh egs/csmsc/voc1/run.sh
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    pushd egs/csmsc/voc1
    dirs -p -v
    bash run.sh --stage -1 --stop-stage 1
    popd
fi
echo "*******prepare benchmark end***********"
