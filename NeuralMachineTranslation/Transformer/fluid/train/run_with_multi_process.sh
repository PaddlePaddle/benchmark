#!/bin/bash
set -xe

export CUDA_VISIBLE_DEVICES=0,1

# Configuration of Allocator and GC
export FLAGS_fraction_of_gpu_memory_to_use=1.0
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_memory_fraction_of_eager_deletion=0.99999

gen_data=/ssd3/transformer_1.1/gen_data
cd ../../../../models/PaddleNLP/neural_machine_translation/transformer/

# base model
python -m paddle.distributed.launch --log_dir=./mylog --selected_gpus=$CUDA_VISIBLE_DEVICES main.py \
	--do_train True \
        --epoch 30 \
        --src_vocab_fpath $gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
        --trg_vocab_fpath $gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
        --special_token '<s>' '<e>' '<unk>' \
	--training_file $gen_data/wmt16_ende_data_bpe/train.tok.clean.bpe.32000.en-de \
	--batch_size 4096

cd -
