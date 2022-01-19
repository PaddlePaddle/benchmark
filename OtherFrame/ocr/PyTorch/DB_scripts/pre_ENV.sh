#!/bin/bash

## 安装依赖

git clone https://github.com/LDOUBLEV/DBNet.pytorch.git
cd DBNet.pytorch
pip3.7 install -r requirement.txt

# 准备数据
bash benchmark/prepare_data.sh

