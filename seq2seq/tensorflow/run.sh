#!/bin/bash

set -xe

export CUDA_VISIBLE_DEVICES=1
export WORK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")" && pwd )"
export BENCHMARK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../.." && pwd )"

task=speed
batch_size=128
model=seq2seq
data_path=${WORK_ROOT}/nmt/data
log_file=${WORK_ROOT}/log_${model}_${task}_bs${batch_size}_1

train() {
  cd $WORK_ROOT/nmt
  rm -rf ./outputs/*
  python -m nmt.nmt \
      --attention=scaled_luong \
      --optimizer adam \
      --learning_rate 0.001 \
      --src=en --tgt=vi \
      --vocab_prefix=${data_path}/vocab  \
      --train_prefix=${data_path}/train \
      --dev_prefix=${data_path}/tst2012  \
      --test_prefix=${data_path}/tst2013 \
      --out_dir=./outputs \
      --batch_size=${batch_size} \
      --num_train_steps=12000 \
      --steps_per_stats=1 \
      --num_layers=2 \
      --num_units=512 \
      --dropout=0.2 \
      --metrics=bleu \
      --num_buckets=5 > ${log_file} 2>&1 &
  train_pid=$!
  sleep 1500
  kill -9 $train_pid
}

analysis() {
  python ${BENCHMARK_ROOT}/scripts/analysis.py \
    --filename ${log_file} \
    --keyword "step-time" \
    --batch_size ${batch_size} \
    --skip_steps 10 \
    --mode 0
}

train
analysis
