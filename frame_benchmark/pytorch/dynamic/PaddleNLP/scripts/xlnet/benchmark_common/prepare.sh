#!/usr/bin/env bash
# 执行路径在模型库的根目录下

################################# 安装框架 如:
echo "*******prepare benchmark start ***********"
pip install -U pip
echo `pip --version`
wget https://paddle-wheel.bj.bcebos.com/benchmark/torch-1.9.1%2Bcu111-cp37-cp37m-linux_x86_64.whl
pip install torch-1.9.1+cu111-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install datasets>=1.8.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install sentencepiece!=0.1.92 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install protobuf -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e .
pip list

#################################
# 运行 train_cmd 能够自动下载训练数据
echo "*******prepare benchmark end***********"
