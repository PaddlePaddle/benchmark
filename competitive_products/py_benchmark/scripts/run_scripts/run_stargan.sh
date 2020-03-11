#!bin/bash

 set -x

if [ $# -lt 2 ]; then
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh train|test speed|mem /ssd2/liyang/logs"
    exit
fi

function _set_params() {
  task="$1"
  index="$2"
  run_log_root=${3:-$(pwd)}
  run_mode="sp"
  skip_steps=2
  keyword="Elapsed"
  separator=" "
  position=17
  range=6
  base_batch_size=16
# the pytorch batch_size is defined in stargan/main.py, and the defalt value is 16
  model_name="StarGAN"
  model_mode=3
  device=${CUDA_VISIBLE_DEVICES//,/ }
  arr=($device)
  num_gpu_devices=${#arr[*]}


  if [[ ${index} = "speed" ]];then
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
  echo "Train on ${num_gpu_devices} GPUs"
  python -u main.py --mode train --dataset CelebA --image_size 128 --c_dim 5 \
      --sample_dir stargan_celeba/samples --log_dir stargan_celeba/logs \
      --model_save_dir stargan_celeba/models --result_dir stargan_celeba/results \
      --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young \
      --n_critic 1 > ${log_file} 2>&1 &
      train_pid=$!
      sleep 300
      kill -9 $train_pid
}

source ${PYTORCH_BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_run
