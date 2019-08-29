#!/bin/bash

set -ex

export CUDA_VISIBLE_DEVICES=0
if [ "${BENCHMAKR_ROOT}" == "" ]; then
  export BENCHMARK_ROOT=/work/benchmark
fi

task=speed
batch_size=128
model=seq2seq
log_file=log_${model}_${task}_bs${batch_size}_1

train() {
  python train.py \
          --src_lang en --tar_lang vi \
          --attention True \
          --num_layers 2 \
          --hidden_size 512 \
          --src_vocab_size 17191 \
          --tar_vocab_size 7709 \
          --batch_size ${batch_size} \
          --dropout 0.2 \
          --init_scale  0.1 \
          --max_grad_norm 5.0 \
          --train_data_prefix data/en-vi/train \
          --eval_data_prefix data/en-vi/tst2012 \
          --test_data_prefix data/en-vi/tst2013 \
          --vocab_prefix data/en-vi/vocab \
          --use_gpu True \
          --max_epoch 2  > ${log_file} 2>&1
}

analysis() {
  python ${BENCHMARK_ROOT}/scripts/analysis.py \
    --filename ${log_file} \
    --keyword "avg_time:" \
    --batch_size ${batch_size} \
    --skip_steps 0 \
    --mode 0
}

train
analysis
