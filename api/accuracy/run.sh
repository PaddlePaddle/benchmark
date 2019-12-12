#!/bin/bash

if [[ $# -lt 1 ]]; then
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 api_name backward[True|False] use_gpu[True|False]"
    echo "Example:"
    echo "  CUDA_VISIBLE_DEVICES=0 bash $0 abs"
    exit
fi

if [ "${CUDA_VISIBLE_DEVICES}" == "" ]; then
    export CUDA_VISIBLE_DEVICES="0"
fi

api_name=$1
backward=${2:-"False"}
use_gpu=${3:-"True"}

python $1.py \
         --backward ${backward} \
         --use_gpu ${use_gpu}
