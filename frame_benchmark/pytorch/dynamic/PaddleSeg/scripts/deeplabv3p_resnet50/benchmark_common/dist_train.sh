#!/usr/bin/env bash

set -e

CONFIG=$1
GPUS=$2
Devices=$3
PORT=${PORT:-29500}

if [[ $Devices -gt 1 ]]; then
    echo "Devices = $Devices"
    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
        python -m torch.distributed.launch --nproc_per_node=$GPUS  --master_port=$PORT \
            --nnodes=$Devices --node_rank=$PADDLE_TRAINER_ID --master_addr=$POD_0_IP $(dirname "$0")/train.py \
            $CONFIG --launcher pytorch ${@:4}
else
    echo "Devices = $Devices"
    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
        python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
            $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:4}
fi