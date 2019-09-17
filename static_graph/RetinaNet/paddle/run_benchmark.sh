#!wqin/bash
set -xe
#python -u tools/train.py -c configs/mask_rcnn_r101_fpn_1x.yml
if [ $# -lt 3 ]; then
  echo "Usage: "
  echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh train speed|mem sp|mp /ssd1/ljh/logs"
  exit
fi


function _set_params(){
  task="$1"
  index="$2"
  run_mode="$3"
  run_log_path=${4:-$(pwd)}
  model_name="cascade_rcnn_fpn"

  device=${CUDA_VISIBLE_DEVICES//,/ }
  arr=($device)
  num_gpu_devices=${#arr[*]}
  base_batch_size=2
  batch_size=`expr ${base_batch_size} \* $num_gpu_devices`
  log_file=${run_log_path}/${model_name}_${task}_${index}_${num_gpu_devices}_${run_mode}
  log_parse_file=${log_file}
}

function _train(){
  echo "Train on ${num_gpu_devices} GPUs"
  echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"

  train_cmd="-c configs/retinanet_r50_fpn_1x.yml"

  case ${run_mode} in
  sp) train_cmd="python -u tools/train.py "${train_cmd} ;;
  mp)
      train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --selected_gpus=$CUDA_VISIBLE_DEVICES tools/train.py "${train_cmd}
      log_parse_file="mylog/workerlog.0" ;;
  *) echo "choose run_mode(sp or mp)"; exit 1;
  esac

  ${train_cmd} > ${log_file} 2>&1 &
  train_pid=$!
  sleep 800
  kill -9 `ps -ef|grep python |awk '{print $2}'`

  if [ $run_mode = "mp" -a -d mylog ]; then
      rm ${log_file}
      cp mylog/workerlog.0 ${log_file}
  fi
}


source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_run

