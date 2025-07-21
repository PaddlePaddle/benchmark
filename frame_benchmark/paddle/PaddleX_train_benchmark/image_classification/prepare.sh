#!/bin/bash
DATASET_BASE_URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/benchmark/dataset"
dataset_url="${DATASET_BASE_URL}/ILSVRC2012_train_benchmark.tar"
tar_name=${dataset_url##*/}
wget -nc $dataset_url -P dataset/
tar -xf dataset/$tar_name -C dataset/
rm -rf dataset/$tar_name