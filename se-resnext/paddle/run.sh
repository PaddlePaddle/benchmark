#!/bin/bash

set -xe


#开启gc
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

#export FLAGS_cudnn_deterministic=true
#export FLAGS_enable_parallel_graph=1

if [ $# -lt 3 ]; then
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh speed|mem|maxbs 32 sp|mp /ssd1/ljh/logs"
    exit
fi

task="train"
index=$1
base_batchsize=$2
run_mode="$3"
run_log_path=${4:-$(pwd)}
model_name="SE-ResNeXt50"

devices_str=${CUDA_VISIBLE_DEVICES//,/ }
gpu_devices=($devices_str)
num_gpu_devices=${#gpu_devices[*]}

if [ $run_mode = "sp" ]; then
    batch_size=`expr $base_batchsize \* $num_gpu_devices`
else
    batch_size=$base_batchsize
fi
cal_batch_size=`expr $base_batchsize \* $num_gpu_devices`

num_epochs=2

log_file=${run_log_path}/${model_name}_${task}_${index}_${num_gpu_devices}_${run_mode}
log_parse_file=${log_file}

train(){
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$cal_batch_size"
    WORK_ROOT=$PWD
    train_cmd=" --model=SE_ResNeXt50_32x4d \
       --batch_size=${batch_size} \
       --total_images=1281167 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --model_save_dir=output/ \
       --pretrained_model=SE_ResNext50_32x4d_pretrained/ \
       --data_dir=data/ILSVRC2012 \
       --with_mem_opt=False \
       --with_inplace=True \
       --lr_strategy=cosine_decay \
       --lr=0.1 \
       --l2_decay=1.2e-4 \
       --num_epochs=${num_epochs}"

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
    kill -9 `ps -ef|grep python |awk '{print $2}'`

    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
    cd ${WORK_ROOT}
}

analysis_times(){
    skip_step=$1
    count_fields=$2
    sed 's/batch\/sec/\ batch\/sec/' ${log_parse_file} | awk 'BEGIN{count=0}/trainbatch/{
      step_times[count]=$'${count_fields}';
      count+=1;
    }END{
      print "\n================ Benchmark Result ================"
      print "num_epochs:", "'${num_epochs}'"
      print "batch_size:", "'${cal_batch_size}'"
      if(count>'${skip_step}'){
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
        printf("average latency (including data reading):\n")
        printf("\tAvg: %.3f s/step\n", step_latency)
        printf("\tFPS: %.3f examples/s\n", "'${cal_batch_size}'"/step_latency)
        printf("average latency (skip '${skip_step}' steps):\n")
        printf("\tAvg: %.3f s/step\n", step_latency_without_step0_avg)
        printf("\tMin: %.3f s/step\n", step_latency_without_step0_min)
        printf("\tMax: %.3f s/step\n", step_latency_without_step0_max)
        printf("\tFPS: %.3f examples/s\n", "'${cal_batch_size}'"/step_latency_without_step0_avg)
        printf("\n")
      }
    }'
}

echo "Benchmark for $index"

if [ ${index} = 'mem' ]
then
    export FLAGS_fraction_of_gpu_memory_to_use=0.001
    gpu_id=`echo $CUDA_VISIBLE_DEVICES |cut -c1`
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
    cpu_num=$(cat /proc/cpuinfo | grep processor | wc -l)
    gpu_num=$(nvidia-smi -L|wc -l)
    awk '{if(NR>1 && $3 >0){time+=$3;count+=1}} END{if(count>0) avg=time/count; else avg=0; printf("avg_gpu_use=%.2f\n" ,avg*'${gpu_num}')}' gpu_avg_utilization
    awk '{if(NR>1 && $3 >0){time+=$3;count+=1}} END{if(count>0) avg=time/count; else avg=0; printf("avg_cpu_use=%.2f\n" ,avg*'${cpu_num}')}' cpu_use
    analysis_times 2 14
else
    train
    error_string="Please shrink FLAGS_fraction_of_gpu_memory_to_use"
    if [ `grep -c "${error_string}" ${log_parse_file}` -eq 0 ]; then
      echo "maxbs is ${batch_size}"
    else
      echo "maxbs running error"
    fi

fi
