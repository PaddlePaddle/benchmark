#!/usr/bin/env bash

# install env
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple

pip install -r requirements.txt

dataset_url="https://paddle-wheel.bj.bcebos.com/benchmark/ILSVRC2012_w.tar"

# prepare data
wget -c ${dataset_url} --no-proxy
tar xf ILSVRC2012_w.tar
rm -f ILSVRC2012_w.tar

echo "*******prepare benchmark end***********"
