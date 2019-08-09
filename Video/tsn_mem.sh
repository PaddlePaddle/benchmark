#!/usr/bin/env bash

# settings
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
model_path="/Paddle/Models"

# flags
export FLAGS_fraction_of_gpu_memory_to_use=0

# paths
log_file='tsn.log'
video_path="${model_path}/PaddleCV/PaddleVideo"

train(){
    dir=$(pwd)
    cd ${video_path}
    train_cmd=" --model_name=TSN \
	--config=${video_path}/configs/tsn.txt \
	--epoch=2 \
	--valid_interval=1 \
	--log_interval=10"
    train_cmd="python3 train.py "${train_cmd}
    ${train_cmd} > ${dir}/${log_file} 2>&1 &
    pid=$!
    sleep 600
    kill -9 $pid
    cd ${dir}
}


analysis_times(){
    skip_step=$1
    awk 'BEGIN{count=0}/speed:/{
      count_fields=NF-1
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
        printf("\tAvg: %.3f steps/s\n", step_latency)
        printf("\tFPS: %.3f examples/s\n", "'${batch_size}'"*step_latency)
        printf("average latency (skip '${skip_step}' steps):\n")
        printf("\tAvg: %.3f steps/s\n", step_latency_without_step0_avg)
        printf("\tMin: %.3f steps/s\n", step_latency_without_step0_min)
        printf("\tMax: %.3f steps/s\n", step_latency_without_step0_max)
        printf("\tFPS: %.3f examples/s\n", '${batch_size}'*step_latency_without_step0_avg)
        printf("\n")
      }
    }' ${log_parse_file}
}

echo "Benchmark for tsn"
gpu_id=`echo $CUDA_VISIBLE_DEVICES | cut -c1`
nvidia-smi --id=$gpu_id --query-compute-apps=used_memory --format=csv -lms 100 > gpu_use.log 2>&1 &
nvidia_pid=$!
train
kill $nvidia_pid
awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "Max=", max}' gpu_use.log

