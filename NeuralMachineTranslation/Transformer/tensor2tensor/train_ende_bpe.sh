export GPU_CARDS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HOME=/home/workspace/t2t_data
PROBLEM=translate_ende_wmt_bpe32k
MODEL=transformer
HPARAMS=transformer_base

DATA_DIR=$HOME/t2t_data_ende16_bpe/
TMP_DIR=$HOME/t2t_datagen_ende16_bpe/ 
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS

#mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR
mkdir -p $TRAIN_DIR

# Generate data
t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM

# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --keep_checkpoint_max=0 \
  --local_eval_frequency=10000 \
  --eval_steps=10000 \
  --train_steps=1000000 \
  --eval_throttle_seconds=8640000 \
  --train_steps=1000000 \
  --worker_gpu=$GPU_CARDS
  --hparams="batch_size=4096"
