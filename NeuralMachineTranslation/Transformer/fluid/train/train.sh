export CUDA_VISIBLE_DEVICES=4,5,6,7 
export FLAGS_enable_parallel_graph=1
export FLAGS_sync_nccl_allreduce=1
gen_data=/ssd1/gs/transformer_1.1/gen_data

# base model
python -u train.py \
  --src_vocab_fpath $gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
  --trg_vocab_fpath $gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
  --special_token '<s>' '<e>' '<unk>' \
  --train_file_pattern ./train.tok.clean.bpe.32000.en-de.tiny \
  --token_delimiter ' ' \
  --use_token_batch True \
  --batch_size 4096 \
  --sort_type pool \
  --pool_size 200000 \
  --shuffle False \
  --enable_ce True \
  --shuffle_batch False \
  --use_py_reader True \
  --use_mem_opt True \
  --fetch_steps 100  $@ \
  dropout_seed 10 \
  learning_rate 2.0 \
  warmup_steps 8000 \
  beta2 0.997 \
  d_model 512 \
  d_inner_hid 2048 \
  n_head 8 \
  prepostprocess_dropout 0.1 \
  attention_dropout 0.1 \
  relu_dropout 0.1 \
  weight_sharing True \
  pass_num 100 \
  model_dir 'tmp_models' \
  ckpt_dir 'tmp_ckpts' 

# big model
# python -u train.py \
#  --src_vocab_fpath $gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
#  --trg_vocab_fpath $gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
#  --special_token '<s>' '<e>' '<unk>' \
#  --train_file_pattern ./train.tok.clean.bpe.32000.en-de.tiny \
#  --token_delimiter ' ' \
#  --use_token_batch True \
#  --batch_size 4096 \
#  --sort_type pool \
#  --pool_size 200000 \
#  --shuffle False \
#  --enable_ce True \
#  --shuffle_batch False \
#  --use_mem_opt True \
#  --use_py_reader True \
#  --enable_ce False \
#  --fetch_steps 100  $@ \
#  learning_rate 2.0 \
#  warmup_steps 8000 \
#  beta2 0.997 \
#  d_model 1024 \
#  d_inner_hid 4096 \
#  n_head 16 \
#  prepostprocess_dropout 0.3 \
#  attention_dropout 0.1 \
#  relu_dropout 0.1 \
#  weight_sharing True \
#  pass_num 100 \
#  max_length 256 \
#  model_dir 'big_trained_models' \
#  ckpt_dir 'big_trained_ckpts'
