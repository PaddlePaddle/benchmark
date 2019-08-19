#!bin/bash

set -xe

if [ $# -lt 3 ]; then
    echo "Usage: "
    echo "CUDA_VISIBLE_DEVICES=0 bash run.sh train speed /ssd3/benchmark_results/cwh/logs"
    exit
fi

export FLAGS_cudnn_exhaustive_search=1
export FLAGS_eager_delete_tensor_gb=0.0

task="$1"
index="$2"
model_name="$3"
run_log_path=${4:-$(pwd)}

device=${CUDA_VISIBLE_DEVICES//,/ }
arr=($device)
num_gpu_devices=${#arr[*]}

batch_size=0
log_file=${run_log_path}/${model_name}_${task}_${index}_${num_gpu_devices}
log_parse_file=${log_file}

train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices"

    if echo {DCGAN CGAN} | grep -w $model_name &>/dev/null
    then
       batch_size=32
       echo "${model_name}, batch_size: ${batch_size}"
       train_cmd=" --model_net $model_name \
           --dataset mnist   \
           --noise_size 100  \
           --batch_size ${batch_size}   \
           --epoch 10"
    elif echo {Pix2pix} | grep -w $model_name &>/dev/null
    then
        batch_size=1
        echo "${model_name}, batch_size: ${batch_size}"
        train_cmd=" --model_net $model_name \
           --dataset cityscapes     \
           --train_list data/cityscapes/pix2pix_train_list \
           --test_list data/cityscapes/pix2pix_test_list    \
           --crop_type Random \
           --dropout True \
           --gan_mode vanilla \
           --batch_size ${batch_size} \
           --epoch 200 \
           --crop_size 256 "
    else
        echo "model: $model_name not support!"
    fi

    train_cmd="python -u train.py "${train_cmd}

    ${train_cmd} > ${log_file} 2>&1 &
    train_pid=$!
    sleep 120
    kill -9 $train_pid
}

analysis_times(){
    skip_step=$1
    awk 'BEGIN{count=0}/Batch_time_cost:/{
      count_fields=NF;
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

if [ $index = "mem" ]
then
    export FLAGS_fraction_of_gpu_memory_to_use=0.001
    gpu_id=`echo $CUDA_VISIBLE_DEVICES | cut -c1`
    nvidia-smi --id=$gpu_id --query-compute-apps=used_memory --format=csv -lms 100 > gpu_use.log 2>&1 &
    gpu_memory_pid=$!
    $task
    kill $gpu_memory_pid
    awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "Max=", max}' gpu_use.log
else
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

    if [ ${task} = "train" ]
    then
      analysis_times 5
    else
      echo "no infer cmd"
    fi
fi
