#!/bin/bash
python3.6 -u train.py \
  --epoch 30 \
  --src_vocab_fpath wmt16_ende_data_bpe_clean/vocab_all.bpe.32000 \
  --trg_vocab_fpath wmt16_ende_data_bpe_clean/vocab_all.bpe.32000 \
  --special_token '<s>' '<e>' '<unk>' \
  --training_file wmt16_ende_data_bpe_clean/train.tok.clean.bpe.32000.en-de \
  --batch_size 4096
