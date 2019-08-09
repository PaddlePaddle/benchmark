#!/usr/bin/env bash

# settings
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
tf_repo_path=/Paddle/youtube-8m
train_data=/Paddle/dataset/youtube8m/tf/train

# args
min=$1
max=$2

# flags

# paths
log_file_prefix='nextvlad_tf-bs'
train_dir=nextvlad_8g_5l2_5drop_128k_2048_2x80_logistic

model_name=NeXtVLADModel
parameters="--groups=8 --nextvlad_cluster_size=128 --nextvlad_hidden_size=2048 \
            --expansion=2 --gating_reduction=8 --drop_rate=0.5"
error_string="ran out of memory"
train(){
    batch_size=$1
    log_file="${log_file_prefix}${batch_size}.log"
    dir=$(pwd)
    cd ${tf_repo_path}
    mkdir ${train_dir}
    train_cmd="python3 train.py \
	${parameters} \
	--model=NeXtVLADModel \
	--num_readers=8 \
	--learning_rate_decay_examples 2000000 \
        --video_level_classifier_model=LogisticModel \
	--label_loss=CrossEntropyLoss \
	--start_new_model=False \
        --train_data_pattern=${train_data}/train*.tfrecord \
	--train_dir=${train_dir} \
	--frame_features=True \
        --feature_names=rgb,audio \
	--feature_sizes=1024,128 \
	--num_epochs=1
	--batch_size=${batch_size} \
	--base_learning_rate=0.0002 \
        --learning_rate_decay=0.8 \
	--l2_penalty=1e-5 \
	--max_step=700000 \
	--num_gpu=2"
    ${train_cmd} > ${dir}/${log_file} 2>&1 &
    pid=$!
    wait $pid
    rm -rf ${train_dir}
    echo "Finish training"
    cd ${dir}
    if [ `grep -c "${error_string}" ${log_file}` -eq 0 ]; then
      return 0
    else
      return 1
    fi
}

while [ $min -lt $max ]; do
    current=`expr '(' "${min}" + "${max}" + 1 ')' / 2`
    echo "Try batchsize=${current}"
    if train ${current}; then
        min=${current}
    else
        max=`expr ${current} - 1`
    fi
done

echo "Maximum batchsize is ${min}"
