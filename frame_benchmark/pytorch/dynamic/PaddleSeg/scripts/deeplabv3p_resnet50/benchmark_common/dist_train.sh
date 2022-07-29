#!/usr/bin/env bash
#将此文件与models/mmsegmentation/tools/dist_train.sh替换
CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#多机
#python3.7 -m torch.distributed.launch --nproc_per_node=$GPUS  --master_port=$PORT \
#    --nnodes=$PADDLE_TRAINERS_NUM --node_rank=$PADDLE_TRAINER_ID --master_addr=$POD_0_IP $(dirname "$0")/train.py \
#    $CONFIG --launcher pytorch ${@:3}
#单机
python3.7 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
