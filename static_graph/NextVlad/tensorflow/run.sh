#!bin/bash

set -x
if [ $# -lt 3 ]; then
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh train|infer speed|mem sp|mp /ssd1/ljh/logs"
    exit
fi

export WORK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")" && pwd )"

datapath=$WORK_ROOT/data/yt8m/train/train
#datapath=./data/yt8m/train/train
train_dir=$WORK_ROOT/nextvlad_8g_5l2_5drop_128k_2048_2x80_logistic
result_folder=$WORK_ROOT/results

function _set_params() {
  task="$1"
  index="$2"
  run_mode="$3"
  model_name="NeXtVLADModel"
  run_log_root=${4:-$(pwd)}
  skip_steps=2
  keyword="Examples/sec:"
  separator=" "
  position=11
  range=5
  model_mode=1 #  # 解析日志，若数据单位是s/step，则为0，若数据单位是step/s,则为1(必填)

  device=${CUDA_VISIBLE_DEVICES//,/ }
  arr=($device)
  num_gpu_devices=${#arr[*]}
  base_batch_size=32
  parameters="--groups=8 --nextvlad_cluster_size=128 --nextvlad_hidden_size=2048 \
            --expansion=2 --gating_reduction=8 --drop_rate=0.5 learning_rate_decay_examples 2 \
            --video_level_classifier_model=LogisticModel --label_loss=CrossEntropyLoss --start_new_model=False \
            --train_data_pattern=${datapath}/train*.tfrecord --train_dir=${train_dir} --frame_features=True \
            --feature_names="rgb,audio" --feature_sizes="1024,128" --base_learning_rate=0.0002 \
            --learning_rate_decay=0.8 --l2_penalty=1e-5"

  if [[ ${index} = "speed" ]]; then
      log_file=${run_log_root}/log_${model_name}_speed_${num_gpu_devices}_${run_mode}
  else
      log_file=${run_log_root}/log_${model_name}_${index}_${num_gpu_devices}_${run_mode}
  fi
  log_parse_file=${log_file}
}

function _set_env() {
  echo "nothing ..."

}

function _train() {
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$base_batch_size"
    WORK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")" && pwd )"
    cd $WORK_ROOT
   # rm -rf $result_folder/*
    parameters="--groups=8 --nextvlad_cluster_size=128 --nextvlad_hidden_size=2048 \
            --expansion=2 --gating_reduction=8 --drop_rate=0.5 learning_rate_decay_examples 2 \
            --video_level_classifier_model=LogisticModel --label_loss=CrossEntropyLoss --start_new_model=False \
            --train_data_pattern=${datapath}/train*.tfrecord --train_dir=${train_dir} --frame_features=True \
            --feature_names="rgb,audio" --feature_sizes="1024,128" --base_learning_rate=0.0002 \
            --learning_rate_decay=0.8 --l2_penalty=1e-5"
     train_cmd=" ${parameters} --model=$model_name \
        --num_readers=8 \
        --num_gpu=$num_gpu_devices \
        --batch_size=$base_batch_size \
        --max_step=700000 \
        --num_epochs=4"

    case ${run_mode} in
    sp)
        train_cmd="python -u train.py "${train_cmd}
        ;;
    mp)
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --selected_gpus=$CUDA_VISIBLE_DEVICES train.py "${train_cmd}
        log_parse_file="mylog/workerlog.0"
        ;;
    *)
        echo "choose run_mode: sp or mp"
        exit 1
        ;;
    esac

    ${train_cmd} > ${log_file} 2>&1 &
    train_pid=$!
    sleep 400
    kill -9 $train_pid
    killall -9 nvidia-smi

    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}
source ${BENCHMARK_ROOT}/competitive_products/common_scripts/run_model.sh
_set_params $@
_set_env
_run
