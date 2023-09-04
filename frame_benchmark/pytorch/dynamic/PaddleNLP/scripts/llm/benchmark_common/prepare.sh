#!/usr/bin/env bash
# 执行路径在模型库的根目录下
################################# 安装框架 如:
echo "*******prepare benchmark start ***********"

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

wget https://bj.bcebos.com/paddlenlp/datasets/examples/llm_benchmark_en.tar.gz
tar -zxvf llm_benchmark_en.tar.gz

wget https://bj.bcebos.com/paddlenlp/datasets/examples/llm_benchmark_zh.tar.gz
tar -zxvf llm_benchmark_zh.tar.gz

echo "*******prepare benchmark end***********"