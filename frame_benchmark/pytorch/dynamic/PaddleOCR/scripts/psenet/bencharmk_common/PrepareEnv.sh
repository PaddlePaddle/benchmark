#!/usr/bin/env bash
# 执行路径在模型库的根目录下
################################# 安装框架 如:
python -m pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install -r requirement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 下载数据集并解压
rm -rf train_data
wget -P ./train_data/ -N https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/icdar2015.tar && cd train_data  && tar xf icdar2015.tar && cd ../
