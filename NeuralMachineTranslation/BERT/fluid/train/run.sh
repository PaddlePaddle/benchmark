#!bin/bash
set -xe

cd ./LARK_Paddle_BERT/BERT/

export FLAGS_cudnn_deterministic=true
export FLAGS_enable_parallel_graph=1
#export FLAGS_eager_delete_tensor_gb=0.0
#export FLAGS_fraction_of_gpu_memory_to_use=0.98
#export FLAGS_memory_fraction_of_eager_deletion=1.0

if [ $# -ne 2 ]; then
  echo "Usage: "
  echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh train|infer speed|mem"
  exit
fi

task="$1"
index="$2"

BERT_BASE_PATH=$(pwd)/../../chinese_L-12_H-768_A-12
TASK_NAME='XNLI'
DATA_PATH=$(pwd)/../../data
CKPT_PATH=$(pwd)/../../save

device=${CUDA_VISIBLE_DEVICES//,/ }
arr=($device)
num_gpu_devices=${#arr[*]}
batch_size=32
log_file=log_${task}_${index}_${num_gpu_devices}

train(){
  echo "Train on ${num_gpu_devices} GPUs"
  echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
  python -u run_classifier.py --task_name ${TASK_NAME} \
       --use_cuda true \
       --do_train true \
       --do_val true \
       --do_test true \
       --batch_size $batch_size \
       --in_tokens False \
       --init_pretraining_params ${BERT_BASE_PATH}/params \
       --data_dir ${DATA_PATH} \
       --vocab_path ${BERT_BASE_PATH}/vocab.txt \
       --checkpoints ${CKPT_PATH} \
       --save_steps 1000 \
       --weight_decay  0.01 \
       --warmup_proportion 0.1 \
       --validation_steps 1000 \
       --epoch 2 \
       --max_seq_len 128 \
       --bert_config_path ${BERT_BASE_PATH}/bert_config.json \
       --learning_rate 5e-5 \
       --skip_steps 100 \
       --random_seed 1 > ${log_file} 2>&1 &
  train_pid=$!
  sleep 600
  kill -9 $train_pid
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
  awk 'BEGIN{count=0}/speed:/{
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
      printf("\tAvg: %.3f steps/s\n", step_latency)
      printf("\tFPS: %.3f examples/s\n", '${batch_size}'*step_latency)
      printf("average latency (skip '${skip_step}' steps):\n")
      printf("\tAvg: %.3f steps/s\n", step_latency_without_step0_avg)
      printf("\tMin: %.3f steps/s\n", step_latency_without_step0_min)
      printf("\tMax: %.3f steps/s\n", step_latency_without_step0_max)
      printf("\tFPS: %.3f examples/s\n", '${batch_size}'*step_latency_without_step0_avg)
      printf("\n")
    }
  }' ${log_file}
}

if [ $index = "mem" ]
then
  echo "Benchmark for $task"
  export FLAGS_fraction_of_gpu_memory_to_use=0.001
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
      analysis_times 1 14
  else
      analysis_times 1 5
  fi
fi

cd -
