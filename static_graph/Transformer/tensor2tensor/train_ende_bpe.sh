#!/bin/bash
set -x

echo "CUDA_VISIBLE_DEVICES=7 bash run_transformer.sh 1|2(speed|mem) big|base /log/path"

function _set_params()
{
    index=$1
    mode=$2
    model_name=transformer_$mode
    run_log_path=${3:-$(pwd)}
    
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    keyword="INFO:tensorflow:global_step/sec:"
    skip_steps=3
    separator=" "
    position=-1
    model_mode=1
    mission_name="机器翻译"           # 模型所属任务名称，具体可参考scripts/config.ini                                （必填）
    direction_id=1 # 任务所属方向，0：CV，1：NLP，2：Rec。                                         (必填)
    run_mode=sp
        
    base_batch_size=4096
    batch_size=`expr ${base_batch_size} \* $num_gpu_devices`
    log_file=${run_log_path}/log_${index}_${model_name}_${num_gpu_devices}
}

function _set_env(){    
    export HOME=/home/workspace/t2t_data
    PROBLEM=translate_ende_wmt_bpe32k
    MODEL=transformer
    HPARAMS=${model_name}
    
    DATA_DIR=$HOME/t2t_data_ende16_bpe/
    TMP_DIR=$HOME/t2t_datagen_ende16_bpe/
    TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS
    
    mkdir -p $TRAIN_DIR
}

function _train(){
    if [ $index -eq 1 ]; then
        sed -i '81c \  config.gpu_options.allow_growth = False' /usr/local/lib/python2.7/dist-packages/tensor2tensor/utils/trainer_lib.py
    elif [ $index -eq 2 ]; then
        echo "this index is: "$index
        sed -i '81c \  config.gpu_options.allow_growth = True' /usr/local/lib/python2.7/dist-packages/tensor2tensor/utils/trainer_lib.py
    fi
    
    rm -rf ${HOME}/*
    #data generate
    t2t-datagen \
       --data_dir=$DATA_DIR \
      --tmp_dir=$TMP_DIR \
      --problem=$PROBLEM > ${log_file} 2>&1
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
      --eval_steps=500 \
      --train_steps=1000 \
      --eval_throttle_seconds=8640000 \
      --train_steps=1000000 \
      --worker_gpu=${num_gpu_devices} \
      --hparams="batch_size=${base_batch_size}" >> ${log_file} 2>&1 &
    
    train_pid=$!
    if [ ${num_gpu_devices} = 1 ]; then
        sleep 800
    elif [ ${num_gpu_devices} = 8 ]; then
        sleep 1100
    else
        sleep 900
    fi
    kill -9 ${train_pid}
}

source ${BENCHMARK_ROOT}/competitive_products/common_scripts/run_model.sh
_set_params $@
_set_env
_run
