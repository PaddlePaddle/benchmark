#!/usr/bin/env bash

# 下载数据
# 安装依赖
if [ -d data ]
then
  rm -rf data
fi

mkdir data && cd data
wget https://paddlenlp.bj.bcebos.com/models/transformers/gpt2/dataset/my-gpt2_text_document.idx
wget https://paddlenlp.bj.bcebos.com/models/transformers/gpt2/dataset/my-gpt2_text_document.bin
cd -

if [ -d token_files ]
then
  rm -rf token_files
fi

mkdir token_files && cd token_files
wget http://paddlenlp.bj.bcebos.com/models/transformers/gpt/gpt-en-vocab.json
wget http://paddlenlp.bj.bcebos.com/models/transformers/gpt/gpt-en-merges.txt
cd -

