#!bin/bash
set -xe

cd ./LARK_Paddle_BERT/BERT/
export CUDA_VISIBLE_DEVICES=0,1,2,3

export FLAGS_cudnn_deterministic=true

export FLAGS_enable_parallel_graph=0

export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=1.0
#export FLAGS_memory_fraction_of_eager_deletion=1.0


BERT_BASE_PATH=$(pwd)/../../chinese_L-12_H-768_A-12
TASK_NAME='XNLI'
DATA_PATH=$(pwd)/../../data
CKPT_PATH=$(pwd)/../../save

device=${CUDA_VISIBLE_DEVICES//,/ }
arr=($device)
num_gpu_devices=${#arr[*]}
if [ $index = "maxbs" ]; then base_batch_size=78; else base_batch_size=32; fi
batch_size=`expr ${base_batch_size} \* $num_gpu_devices`
log_file=${run_log_path}/${model_name}_${task}_${index}_${num_gpu_devices}

python -m paddle.distributed.launch --log_dir=./mylog --selected_gpus=$CUDA_VISIBLE_DEVICES run_classifier.py --task_name ${TASK_NAME} \
       --use_cuda true \
       --do_train true \
       --do_val false \
       --do_test false \
       --batch_size ${base_batch_size} \
       --in_tokens False \
       --init_pretraining_params ${BERT_BASE_PATH}/params \
       --data_dir ${DATA_PATH} \
       --vocab_path ${BERT_BASE_PATH}/vocab.txt \
       --checkpoints ${CKPT_PATH} \
       --save_steps 1000 \
       --weight_decay  0.01 \
       --warmup_proportion 0.1 \
       --validation_steps 1000 \
       --epoch 2 \
       --max_seq_len 128 \
       --bert_config_path ${BERT_BASE_PATH}/bert_config.json \
       --learning_rate 5e-5 \
       --skip_steps 100 \
       --random_seed 1

