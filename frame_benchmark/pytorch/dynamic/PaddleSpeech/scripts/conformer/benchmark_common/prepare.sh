#!/usr/bin/env bash
# 执行路径在模型库的根目录下
################################# 安装框架 如:
echo "*******prepare benchmark start ***********"
data_tiny_url="None"

if [ ${data_tiny_url} == "None" ]; then
	echo "please contact the author to get the URL"
	exit
fi

set -e
# 修改竞品的训练日志
cp replace/executor.py wenet/utils/
cp replace/run.sh examples/aishell/s0
cp replace/download_and_untar.sh examples/aishell/s0/local

#pip install -U pip
#echo `pip --version`

test -d venv || virtualenv -p python venv
source venv/bin/activate
# pip install torch==1.9.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
#conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip list

# wget https://paddleseg.bj.bcebos.com/benchmark/mmseg/mmseg_benchmark_configs.tar.gz
# tar -zxf mmseg_benchmark_configs.tar.gz
################################# 准备训练数据 如:

cd examples/aishell/s0
mkdir -p data_store
bash run.sh --stage -1 --stop_stage 3 --data_tiny_url ${data_tiny_url}
cd -

VERSION=v4.2.0
BINARY=yq_linux_amd64
wget https://github.com/mikefarah/yq/releases/download/${VERSION}/${BINARY} -O /usr/bin/yq &&\
        chmod +x /usr/bin/yq

echo "*******prepare benchmark end***********"
