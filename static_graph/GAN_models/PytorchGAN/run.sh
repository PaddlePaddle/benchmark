#!bin/bash

set -xe

if [ $# -lt 1 ]; then
    echo "Usage: "
    echo "CUDA_VISIBLE_DEVICES=0 bash run.sh train /ssd3/benchmark_results/cwh/logs"
    exit
fi

task="$1"
run_log_path=${2:-$(pwd)}

device=${CUDA_VISIBLE_DEVICES//,/ }
arr=($device)
num_gpu_devices=${#arr[*]}

batch_size=1
log_file=${run_log_path}/Pytorch_Pix2Pix_GPU_cards_${num_gpu_devices}
log_parse_file=${log_file}

train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices"

    env no_proxy=localhost  python3.5 train.py  \
              --dataroot ./datasets/cityscapes  \
              --name cityscapes_pix2pix         \
              --model pix2pix     \
              --netG unet_256     \
              --direction BtoA    \
              --lambda_L1 100     \
              --dataset_mode aligned \
              --norm batch        \
              --pool_size 0   > ${log_file} 2>&1 &

    train_pid=$!
    sleep 200
    kill -9 $train_pid
}

analysis_times(){
    skip_step=$1
    awk 'BEGIN{count=0}/time:/{
      count_fields=6;
      step_times[count]=$count_fields;
      count+=1;
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
          printf("%.5f", step_times[i])
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
        printf("\tFPS: %.3f images/s\n", "'${batch_size}'"/step_latency)
        printf("average latency (skip '${skip_step}' steps):\n")
        printf("\tAvg: %.3f s/step\n", step_latency_without_step0_avg)
        printf("\tMin: %.3f s/step\n", step_latency_without_step0_min)
        printf("\tMax: %.3f s/step\n", step_latency_without_step0_max)
        printf("\tFPS: %.3f images/s\n", '${batch_size}'/step_latency_without_step0_avg)
        printf("\n")
      }
    }' ${log_parse_file}
}

echo "Benchmark for $task"

echo "test speed"
job_bt=`date '+%Y%m%d%H%M%S'`
$task
job_et=`date '+%Y%m%d%H%M%S'`
hostname=`echo $(hostname)|awk -F '.baidu.com' '{print $1}'`
# monquery -n $hostname -i GPU_AVERAGE_UTILIZATION -s $job_bt -e $job_et -d 60 > gpu_avg_utilization
# monquery -n $hostname -i CPU_USER -s $job_bt -e $job_et -d 60 > cpu_use
cpu_num=$(cat /proc/cpuinfo | grep processor | wc -l)
gpu_num=$(nvidia-smi -L|wc -l)
# awk '{if(NR>1 && $3 >0){time+=$3;count+=1}} END{if(count>0) avg=time/count; else avg=0; printf("avg_gpu_use=%.2f\n" ,avg*'${gpu_num}')}' gpu_avg_utilization
# awk '{if(NR>1 && $3 >0){time+=$3;count+=1}} END{if(count>0) avg=time/count; else avg=0; printf("avg_cpu_use=%.2f\n" ,avg*'${cpu_num}')}' cpu_use

analysis_times 5
