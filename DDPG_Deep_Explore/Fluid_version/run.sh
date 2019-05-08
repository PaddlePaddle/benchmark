#!bin/bash
set -xe

if [ $# -ne 2 ]; then
  echo "Usage: "
  echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh train|infer speed|mem"
  exit
fi

#打开后速度变快
export FLAGS_cudnn_exhaustive_search=1

#开启
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fast_eager_deletion_mode=1

task="$1"
index="$2"

device=${CUDA_VISIBLE_DEVICES//,/ }
arr=($device)
num_gpu_devices=${#arr[*]}
batch_size=1
log_file=log_${task}_${index}_${num_gpu_devices}

train(){
  echo "Train on ${num_gpu_devices} GPUs"
  echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
  for i in {1..5}
  do
      FLAGS_enforce_when_check_program_=0 GLOG_vmodule=operator=1,computation_op_handle=1 \
      python ./multi_thread_test.py \
          --ensemble_num 1 \
          --test_times 10 >> ${log_file} 2>&1
  done
}

infer(){
  echo "infer on ${num_gpu_devices} GPUs"
  echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
#  python eval_coco_map.py \
#    --dataset=coco2017 \
#    --pretrained_model=../imagenet_resnet50_fusebn/ \
#    --MASK_ON=True > ${log_file} 2>&1 &
#  infer_pid=$!
#  sleep 60
#  kill -9 $infer_pid
}

analysis_times(){
  skip_step=$1
  count_fields=$2
  awk 'BEGIN{count=0}/time consuming:/{
    step_times[count]=$'${count_fields}';
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
      printf("\tAvg: %.3f s/epoch\n", step_latency)
      printf("\tFPS: %.3f examples/s\n", "'${batch_size}'"/step_latency)
      printf("average latency (skip '${skip_step}' steps):\n")
      printf("\tAvg: %.3f s/epoch\n", step_latency_without_step0_avg)
      printf("\tMin: %.3f s/epoch\n", step_latency_without_step0_min)
      printf("\tMax: %.3f s/epoch\n", step_latency_without_step0_max)
      printf("\tFPS: %.3f examples/s\n", '${batch_size}'/step_latency_without_step0_avg)
      printf("\n")
    }
  }' ${log_file}
}

if [ $index = "mem" ]
then
    echo "Benchmark for $task"
    #若测试最大batchsize，FLAGS_fraction_of_gpu_memory_to_use=1
    export FLAGS_fraction_of_gpu_memory_to_use=0.001
#    export FLAGS_enable_parallel_graph=0
#    export FLAGS_memory_fraction_of_eager_deletion=0.99999
#    export FLAGS_eager_delete_tensor_gb=0.0
    gpu_id=`echo $CUDA_VISIBLE_DEVICES | cut -c1`
    nvidia-smi --id=$gpu_id --query-compute-apps=used_memory --format=csv -lms 100 > gpu_use.log 2>&1 &
    gpu_memory_pid=$!
    $task
    kill $gpu_memory_pid
    awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "Max=", max}' gpu_use.log
else
    echo "Benchmark for $task"

    $task
    if [ ${task} = "train" ]
    then
      analysis_times 1 10
    else
      echo "no infer cmd"
      #analysis_times 3 5 5
    fi
fi

#source activate python35
#export CUDA_VISIBLE_DEVICES="1"

#wget ftp://yq01-sys-hic-p40-box-a12-0057.yq01.baidu.com:/home/users/minqiyang/workspace/paddle/Paddle/build935/accelerate_ddpg/python/dist/paddlepaddle_gpu-0.0.0-cp35-cp35m-linux_x86_64.whl -O paddlepaddle_gpu-0.0.0-cp35-cp35m-linux_x86_64.whl && pip uninstall -y paddlepaddle-gpu && pip install paddlepaddle_gpu-0.0.0-cp35-cp35m-linux_x86_64.whl

#export PATH=/usr/local/cuda/bin:$PATH
#FLAGS_enforce_when_check_program_=0 GLOG_vmodule=operator=1,computation_op_handle=1 python ./multi_thread_test.py --ensemble_num 1 --test_times 10 >log 2>errorlog

#python timeline.py --profile_path=./profile --timeline_path=./timeline