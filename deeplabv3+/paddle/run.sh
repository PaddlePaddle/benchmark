#!/bin/bash

set -x

#export FLAGS_cudnn_deterministic=true
#export FLAGS_enable_parallel_graph=1

export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fast_eager_deletion_mode=1

if [ $# -lt 3 ]; then
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh train|infer speed|mem|maxbs sp|mp /ssd1/ljh/logs"
    exit
fi

task="$1"
index="$2"
run_mode="$3"
run_log_path=${4:-$(pwd)}
model_name="DeepLab_V3+"

DATASET_PATH=${PWD}/data/cityscape/
INIT_WEIGHTS_PATH=${PWD}/deeplabv3plus_xception65_initialize
SAVE_WEIGHTS_PATH=${PWD}/output/model
echo $DATASET_PATH

devices_str=${CUDA_VISIBLE_DEVICES//,/ }
gpu_devices=($devices_str)
num_gpu_devices=${#gpu_devices[*]}

train_crop_size=513
total_step=80
if [ $index = "maxbs" ]; then base_batch_size=9; else base_batch_size=2; fi
batch_size=`expr ${base_batch_size} \* $num_gpu_devices`

log_file=${run_log_path}/${model_name}_${task}_${index}_${num_gpu_devices}
log_parse_file=${log_file}

train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"

    train_cmd=" --batch_size=${batch_size} \
        --train_crop_size=${train_crop_size} \
        --total_step=${total_step} \
        --init_weights_path=${INIT_WEIGHTS_PATH} \
        --save_weights_path=${SAVE_WEIGHTS_PATH} \
        --dataset_path=${DATASET_PATH} \
        --parallel=True"

    case ${run_mode} in
    sp) train_cmd="python -u train.py "${train_cmd} ;;
    mp)
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --selected_gpus=${CUDA_VISIBLE_DEVICES} train.py "${train_cmd}
        log_parse_file="mylog/workerlog.0" ;;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac

    ${train_cmd} > ${log_file} 2>&1
    # Python multi-processing is used to read images, so need to
    # kill those processes if the main train process is aborted.
    #ps -aux | grep "$PWD/train.py" | awk '{print $2}' | xargs kill -9
    kill -9 `ps -ef|grep 'deeplabv3+'|awk '{print $2}'`

    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}
analysis_times(){
    awk 'BEGIN{count=0}/step_time_cost:/{
      step_times[count]=$6;
      count+=1;
    }END{
      print "\n================ Benchmark Result ================"
      print "total_step:", "'${total_step}'"
      print "batch_size:", "'${batch_size}'"
      print "train_crop_size:", "'${train_crop_size}'"
      if(count>1){
        step_latency=0
        step_latency_without_step0_avg=0
        step_latency_without_step0_min=step_times[1]
        step_latency_without_step0_max=step_times[1]
        for(i=0;i<count;++i){
          step_latency+=step_times[i];
          if(i>0){
            step_latency_without_step0_avg+=step_times[i];
            if(step_times[i]<step_latency_without_step0_min){
              step_latency_without_step0_min=step_times[i];
            }
            if(step_times[i]>step_latency_without_step0_max){
              step_latency_without_step0_max=step_times[i];
            }
          }
        }
        step_latency/=count;
        step_latency_without_step0_avg/=(count-1)
        printf("average latency (including data reading):\n")
        printf("\tAvg: %.3f s/step\n", step_latency)
        printf("\tFPS: %.3f examples/s\n", "'${batch_size}'"/step_latency)
        printf("average latency (including data reading, without step 0):\n")
        printf("\tAvg: %.3f s/step\n", step_latency_without_step0_avg)
        printf("\tMin: %.3f s/step\n", step_latency_without_step0_min)
        printf("\tMax: %.3f s/step\n", step_latency_without_step0_max)
        printf("\tFPS: %.3f examples/s\n", "'${batch_size}'"/step_latency_without_step0_avg)
        printf("\n")
      }
    }' ${log_parse_file}
}

echo "Benchmark for $index"

if [ $index = "mem" ]
then
    export FLAGS_fraction_of_gpu_memory_to_use=0.001
    gpu_id=`echo $CUDA_VISIBLE_DEVICES | cut -c1`
    nvidia-smi --id=$gpu_id --query-compute-apps=used_memory --format=csv -lms 100 > gpu_use.log 2>&1 &
    gpu_memory_pid=$!
    train
    kill $gpu_memory_pid
    awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "Max=", max}' gpu_use.log
elif [ ${index} = 'speed' ]
then
    job_bt=`date '+%Y%m%d%H%M%S'`
    train
    job_et=`date '+%Y%m%d%H%M%S'`
    hostname=`echo $(hostname)|awk -F '.baidu.com' '{print $1}'`
    monquery -n $hostname -i GPU_AVERAGE_UTILIZATION -s $job_bt -e $job_et -d 60 > gpu_avg_utilization
    monquery -n $hostname -i CPU_USER -s $job_bt -e $job_et -d 60 > cpu_use
    awk '{if(NR>1){time+=$3;count+=1}} END{if(count>0) avg=time/count; else avg=0; printf("avg_gpu_use=%.2f\n" ,avg)}' gpu_avg_utilization
    awk '{if(NR>1){time+=$3;count+=1}} END{if(count>0) avg=time/count; else avg=0; printf("avg_cpu_use=%.2f\n" ,avg)}' cpu_use
    analysis_times
else
    train
    error_string="Please shrink FLAGS_fraction_of_gpu_memory_to_use or FLAGS_initial_gpu_memory_in_mb or FLAGS_reallocate_gpu_memory_in_mbenvironment variable to a lower value"
    if [ `grep -c "${error_string}" ${log_parse_file}` -eq 0 ]; then
      echo "maxbs is ${batch_size}"
    else
      echo "maxbs running error"
    fi
fi
