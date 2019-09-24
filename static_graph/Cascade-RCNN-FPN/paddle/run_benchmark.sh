#!bin/bash
set -xe
if [[ $# -lt 4 ]]; then
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh speed|mem|maxbs 2 model_name sp|mp /ssd1/ljh/logs"
    exit
fi

function _set_params(){
    index=$1                         # 速度(speed)|显存占用(mem)|单卡最大支持batch_size(maxbs)(必填)
    base_batch_size=$2               # 单卡的batch_size，如果固定的，可以写死（必填）
    model_name=$3                    # 模型名字如："SE-ResNeXt50"，如果是固定的，可以写死，如果需要其他参数可以参考bert实现（必填）
    run_mode=${4:-"sp"}              # 单进程(sp)|多进程(mp)，默认单进程（必填）
    run_log_path=${5:-$(pwd)}        # 训练保存的日志目录（必填）

    skip_steps=2                     # 解析日志，有些模型前几个step耗时长，需要跳过(必填)
    keyword="iter"                   # 解析日志，筛选出数据所在行的关键字(必填)
    separator=" "                    # 解析日志，数据所在行的分隔符(必填)
    position=25                      # 解析日志，按照分隔符分割后形成的数组索引(必填)
    model_mode=0                     # 解析日志，若数据单位是s/step，则为0，若数据单位是step/s,则为1(必填)
    range=-1
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    if [ $run_mode = "sp" ]; then
        batch_size=`expr $base_batch_size \* $num_gpu_devices`
    else
        batch_size=$base_batch_size
    fi
    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_parse_file=${log_file}

}

function _set_env(){
    #开启gc
    export FLAGS_eager_delete_tensor_gb=0.0
    export FLAGS_fraction_of_gpu_memory_to_use=0.98
}

function _train(){
  echo "Train on ${num_gpu_devices} GPUs"
  echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"

  train_cmd="-c configs/cascade_rcnn_r50_fpn_1x.yml"

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
_set_env
_run

