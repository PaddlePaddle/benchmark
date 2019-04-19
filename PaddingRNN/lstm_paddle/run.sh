#!/bin/sh

if [ $# -lt 2 ]; then
  batch_size=20
else
  batch_size=$2
fi

if [ $# -lt 1 ]; then
  model_type=large
else
  model_type=$1
fi

# Occupy all GPU memory (5% reserved actually)
export FLAGS_fraction_of_gpu_memory_to_use=1.0

# Enable gc when this flag is larger than or equal to 0. 
# If you change this value, please make sure that the large
# model can run when batch_size = 2100; and the small model
# can run when batch_size = 5365
export FLAGS_eager_delete_tensor_gb=0.0

# You can set this ratio to control the number of gc ops 
# GC is disabled when this flag is 0; and full gc would be
# performed when this flag is 1. Must be inside [0, 1].
# If you change this value, please make sure that the large
# model can run when batch_size = 2100; and the small model
# can run when batch_size = 5365
export FLAGS_memory_fraction_of_eager_deletion=1.0

python train.py --data_path data/simple-examples/data/ --model_type $model_type --use_gpu True --batch_size $batch_size
