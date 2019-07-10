#!/bin/bash
set -xe

#export FLAGS_cudnn_deterministic=true
#export FLAGS_enable_parallel_graph=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98
export FLAGS_memory_fraction_of_eager_deletion=1.0

if [ $# -lt 3 ]; then
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh train|infer speed|mem|maxbs sp|mp /ssd1/ljh/logs"
    exit
fi

task="$1"
index="$2"
run_mode="$3"
run_log_path=${4:-$(pwd)}
model_name="yolov3"

device=${CUDA_VISIBLE_DEVICES//,/ }
arr=($device)
num_gpu_devices=${#arr[*]}
if [ $index = "maxbs" ]; then base_batch_size=14; else base_batch_size=8; fi
batch_size=`expr ${base_batch_size} \* $num_gpu_devices`
log_file=${run_log_path}/${model_name}_${task}_${index}_${num_gpu_devices}_${run_mode}
log_parse_file=${log_file}

train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"

    train_cmd=" --model_save_dir=output/ \
     --pretrain=./weights/darknet53/ \
     --data_dir=./dataset/coco/ \
     --batch_size=${base_batch_size} \
     --syncbn=False"

    case ${run_mode} in
    sp) train_cmd="python -u train.py "${train_cmd} ;;
    mp)
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --selected_gpus=$CUDA_VISIBLE_DEVICES train.py "${train_cmd}
        log_parse_file="mylog/workerlog.0" ;;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac

    ${train_cmd} > ${log_file} 2>&1 &
    train_pid=$!
    sleep 600
    #kill -9 $train_pid
    kill -9 `ps -ef|grep 'python'|awk '{print $2}'`

    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}

infer(){
    echo "infer on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
    python eval.py \
      --data_dir=./dataset/coco/ \
      --batch_size=$batch_size \
      --weights=./weights/yolov3 > ${log_file} 2>&1 &
    infer_pid=$!
    sleep 60
    kill -9 $infer_pid
}

analysis_times(){
    skip_step=$1
    filter_fields=$2
    count_fields=$3
    awk 'BEGIN{count=0}{
      if(NF=='${filter_fields}'){
      step_times[count]=$'${count_fields}';
      count+=1;}
    }END{
      print "\n================ Benchmark Result ================"
      print "total_step:", count
      print "batch_size:", "'${batch_size}'"
      if(count>1){
        step_latency=0
        step_latency_without_step0_avg=0
        step_latency_without_step0_min=step_times['${skip_step}']
        step_latency_without_step0_max=step_times['${skip_step}']
        for(i=0;i<count;++i){
          step_latency+=step_times[i];
          if(i>='${skip_step}'){
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
        step_latency_without_step0_avg/=(count-'${skip_step}')
        printf("average latency (origin result):\n")
        printf("\tAvg: %.3f s/step\n", step_latency)
        printf("\tFPS: %.3f examples/s\n", "'${batch_size}'"/step_latency)
        printf("average latency (skip '${skip_step}' steps):\n")
        printf("\tAvg: %.3f s/step\n", step_latency_without_step0_avg)
        printf("\tMin: %.3f s/step\n", step_latency_without_step0_min)
        printf("\tMax: %.3f s/step\n", step_latency_without_step0_max)
        printf("\tFPS: %.3f examples/s\n", '${batch_size}'/step_latency_without_step0_avg)
        printf("\n")
      }
    }' ${log_parse_file}
}

echo "Benchmark for $index"

if [ ${index} = "mem" ]
then
    export FLAGS_fraction_of_gpu_memory_to_use=0.001
    gpu_id=`echo $CUDA_VISIBLE_DEVICES | cut -c1`
    nvidia-smi --id=$gpu_id --query-compute-apps=used_memory --format=csv -lms 100 > gpu_use.log 2>&1 &
    gpu_memory_pid=$!
    $task
    kill $gpu_memory_pid
    awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "Max=", max}' gpu_use.log
elif [ ${index} = 'speed' ]
then
    job_bt=`date '+%Y%m%d%H%M%S'`
    $task
    job_et=`date '+%Y%m%d%H%M%S'`
    hostname=`echo $(hostname)|awk -F '.baidu.com' '{print $1}'`
    monquery -n $hostname -i GPU_AVERAGE_UTILIZATION -s $job_bt -e $job_et -d 60 > gpu_avg_utilization
    monquery -n $hostname -i CPU_USER -s $job_bt -e $job_et -d 60 > cpu_use
    awk '{if(NR>1 && $3 >0){time+=$3;count+=1}} END{if(count>0) avg=time/count; else avg=0; printf("avg_gpu_use=%.2f\n" ,avg)}' gpu_avg_utilization
    awk '{if(NR>1 && $3 >0){time+=$3;count+=1}} END{if(count>0) avg=time/count; else avg=0; printf("avg_cpu_use=%.2f\n" ,avg)}' cpu_use
    if [ ${task} = "train" ]
    then
        analysis_times 5 8 8
    else
        analysis_times 3 5 5
    fi
else
    $task
    error_string="Please shrink FLAGS_fraction_of_gpu_memory_to_use or FLAGS_initial_gpu_memory_in_mb or FLAGS_reallocate_gpu_memory_in_mbenvironment variable to a lower value"
    if [ `grep -c "${error_string}" ${log_parse_file}` -eq 0 ]; then
      echo "maxbs is ${batch_size}"
    else
      echo "maxbs running error"
    fi
fi
