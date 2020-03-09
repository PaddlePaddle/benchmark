#!/bin/bash
python3.6 -u train.py \
  --epoch 30 \
  --src_vocab_fpath gen_data/iwslt14.tokenized.de-en/vocab.de \
  --trg_vocab_fpath gen_data/iwslt14.tokenized.de-en/vocab.en \
  --special_token '<s>' '<e>' '<unk>' \
  --training_file gen_data/iwslt14.tokenized.de-en/para_small.de-en \
  --batch_size 4096
