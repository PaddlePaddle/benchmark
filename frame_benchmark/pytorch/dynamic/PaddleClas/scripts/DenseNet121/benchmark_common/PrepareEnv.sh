#!/usr/bin/env bash

# install env
wget -c --no-proxy ${FLAG_TORCH_WHL_URL}
tar_file_name=$(echo ${FLAG_TORCH_WHL_URL} | awk -F '/' '{print $NF}')
dir_name=$(echo ${tar_file_name} | awk -F '.' '{print $1}')
tar xf ${tar_file_name}
rm -rf ${tar_file_name}

pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
pip install ${dir_name}/*
pip install -r requirements.txt
# fix bug to adapt to torch 2.0
sed -i 's/local_rank/local-rank/g' train.py


# prepare data
dataset_url="https://paddleclas.bj.bcebos.com/data/TIPC/ILSVRC2012_benchmark.tar"
wget -c ${dataset_url} --no-proxy
tar xf ILSVRC2012_benchmark.tar
mv ILSVRC2012_benchmark ILSVRC2012_w
rm -f ILSVRC2012_benchmark.tar

echo "*******prepare benchmark end***********"
