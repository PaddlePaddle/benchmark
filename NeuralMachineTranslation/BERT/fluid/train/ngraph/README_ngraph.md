# PaddlePaddle BERT Training with nGraph

To run Bert training:

```
./train.sh
```

The script will do the following:

1. git clone paddlepaddle/LARK. The model is from https://github.com/PaddlePaddle/LARK/tree/develop/BERT
2. Set env exports for nGraph
```
export OMP_NUM_THREADS=<num_cpu_cores>
export KMP_AFFINITY=granularity=fine,compact,1,0
```
The KMP_AFFINITTY is recommended if multiple threads are used.
3. Run the bert training (Paddlepaddle needs to be installed)
```
FLAGS_use_ngraph=true python -u ./train.py \
        --is_distributed false\
        --use_cuda false\
        --weight_sharing true\
        --batch_size ${BATCH_SIZE} \
        --data_dir ${TRAIN_DATA_DIR} \
        --validation_set_dir ${VALIDATION_DATA_DIR} \
        --bert_config_path ${CONFIG_PATH} \
        --vocab_path ${VOCAB_PATH} \
        --generate_neg_sample true\
        --checkpoints ./output \
        --save_steps ${SAVE_STEPS} \
        --lr_scheduler ${LR_SCHEDULER} \
        --learning_rate ${LR_RATE} \
        --weight_decay ${WEIGHT_DECAY:-0} \
        --max_seq_len ${MAX_LEN} \
        --skip_steps 20 \
        --validation_steps 1000 \
        --num_iteration_per_drop_scope 10 \
        --in_token false\
        --use_fp16 false \
        --loss_scaling 8.0
```
Please use `noam_decay` for `lr_scheduler` and set `in_token` as false.
