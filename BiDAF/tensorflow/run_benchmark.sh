#!bin/bash
set -xe

if [ $# -lt 2 ]; then
  echo "Usage: "
  echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh train|infer speed|mem|maxbs /ssd1/ljh/logs"
  exit
fi

#export FLAGS_enable_parallel_graph=0
#export FLAGS_sync_nccl_allreduce=1

# Configuration of Allocator and GC
#export FLAGS_fraction_of_gpu_memory_to_use=1.0
#export FLAGS_eager_delete_tensor_gb=0.0
#export FLAGS_memory_fraction_of_eager_deletion=0.99999

task="$1"
index="$2"
run_log_path=${3:-$(pwd)}
model_name="BiDAF"

device=${CUDA_VISIBLE_DEVICES//,/ }
arr=($device)
num_gpu_devices=${#arr[*]}
batch_size=32
log_file=${run_log_path}/${model_name}_${task}_${index}_${num_gpu_devices}

train(){
  echo "Train on ${num_gpu_devices} GPUs"
  echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
  # base model
  train_cmd="python run.py --train --algo BIDAF --epochs 10 \
    --train_files  ../data/extracted/trainset/zhidao.train.json ../data/extracted/trainset/search.train.json \
    --dev_files ../data/extracted/devset/zhidao.dev.json ../data/extracted/devset/search.dev.json \
    --test_files  ../data/extracted/testset/zhidao.test.json ../data/extracted/testset/search.test.json"

  ${train_cmd} > ${log_file} 2>&1 &
  train_pid=$!
  sleep 1800
  kill -9 `ps -ef|grep python |awk '{print $2}'`
}

infer(){
  echo "infer on ${num_gpu_devices} GPUs"
  echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
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
  }' ${log_file}
}

echo "Benchmark for $task"

if [ $index = "mem" ]
then
    if [ `grep -c "sess_config.gpu_options.allow_growth" rc_model.py` -gt 0 ]; then
      echo "Found!"
    else
      sed -i '/sess_config = tf.ConfigProto()/a\        sess_config.gpu_options.allow_growth = True' rc_model.py
    fi

    gpu_id=`echo $CUDA_VISIBLE_DEVICES | cut -c1`
    nvidia-smi --id=$gpu_id --query-compute-apps=used_memory --format=csv -lms 100 > gpu_use.log 2>&1 &
    gpu_memory_pid=$!
    $task
    kill $gpu_memory_pid
    awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "Max=", max}' gpu_use.log
elif [ ${index} = 'speed' ]
then
    sed -i '/sess_config.gpu_options.allow_growth/d' rc_model.py
    $task
    if [ ${task} = "train" ]
    then
      analysis_times 0
    else
      echo "no infer cmd"
      #analysis_times 3
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