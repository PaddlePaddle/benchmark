#!bin/bash
set -xe

if [ $# -lt 3 ]; then
  echo "Usage: "
  echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh train|infer speed|mem|maxbs sp|mp /ssd1/ljh/logs"
  exit
fi

export FLAGS_enable_parallel_graph=0
export FLAGS_sync_nccl_allreduce=1

# Configuration of Allocator and GC
export FLAGS_fraction_of_gpu_memory_to_use=1.0
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_memory_fraction_of_eager_deletion=0.99999

task="$1"
index="$2"
run_mode="$3"
run_log_path=${4:-$(pwd)}
model_name="transformer"

device=${CUDA_VISIBLE_DEVICES//,/ }
arr=($device)
num_gpu_devices=${#arr[*]}
if [ $index = "maxbs" ]; then base_batch_size=12000; else base_batch_size=4096; fi
batch_size=`expr ${base_batch_size} \* $num_gpu_devices`
log_file=${run_log_path}/${model_name}_${task}_${index}_${num_gpu_devices}
log_parse_file=${log_file}

train(){
  echo "Train on ${num_gpu_devices} GPUs"
  echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
  # base model
  train_cmd=" --src_vocab_fpath data/vocab.bpe.32000 \
      --trg_vocab_fpath data/vocab.bpe.32000 \
      --special_token <s> <e> <unk> \
      --train_file_pattern data/train.tok.clean.bpe.32000.en-de \
      --use_token_batch True \
      --batch_size ${base_batch_size} \
      --sort_type pool \
      --pool_size 200000 \
      --shuffle False \
      --enable_ce True \
      --shuffle_batch False \
      --use_py_reader True \
      --use_mem_opt True \
      --use_default_pe False \
      --fetch_steps 100  $@ \
      dropout_seed 10 \
      learning_rate 2.0 \
      warmup_steps 8000 \
      beta2 0.997 \
      d_model 512 \
      d_inner_hid 2048 \
      n_head 8 \
      prepostprocess_dropout 0.1 \
      attention_dropout 0.1 \
      relu_dropout 0.1 \
      weight_sharing True \
      pass_num 1 \
      model_dir tmp_models \
      ckpt_dir tmp_ckpts"

  case ${run_mode} in
  sp) train_cmd="python -u train.py "${train_cmd} ;;
  mp)
      train_cmd="python -m paddle.distributed.launch --gpus ${num_gpu_devices}  train.py "${train_cmd}
      log_parse_file="mylog/workerlog.0" ;;
  *) echo "choose run_mode(sp or mp)"; exit 1;
  esac

  ${train_cmd} > ${log_file} 2>&1 &
  train_pid=$!
  sleep 900
  kill -9 `ps -ef|grep python |awk '{print $2}'`
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
  filter_fields=$2
  count_fields=$3
  awk 'BEGIN{count=0}{if(NF=='${filter_fields}'){
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

echo "Benchmark for $task"

if [ $index = "mem" ]
then
    #若测试最大batchsize，FLAGS_fraction_of_gpu_memory_to_use=1
    export FLAGS_fraction_of_gpu_memory_to_use=0.001
    gpu_id=`echo $CUDA_VISIBLE_DEVICES | cut -c1`
    nvidia-smi --id=$gpu_id --query-compute-apps=used_memory --format=csv -lms 100 > gpu_use.log 2>&1 &
    gpu_memory_pid=$!
    $task
    kill $gpu_memory_pid
    awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "Max=", max}' gpu_use.log
elif [ ${index} = 'speed' ]
then
    $task
    if [ ${task} = "train" ]
    then
      analysis_times 3 19 18
    else
      echo "no infer cmd"
      #analysis_times 3 5 5
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