#!/usr/bin/env bash

# settings
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
tf_repo_path=/Paddle/youtube-8m
train_data=/Paddle/dataset/youtube8m/tf/train

# flags

# paths
log_file='nextvlad_tf.log'
train_dir=nextvlad_8g_5l2_5drop_128k_2048_2x80_logistic

model_name=NeXtVLADModel
parameters="--groups=8 --nextvlad_cluster_size=128 --nextvlad_hidden_size=2048 \
            --expansion=2 --gating_reduction=8 --drop_rate=0.5"
train(){
    dir=$(pwd)
    cd ${tf_repo_path}
    mkdir ${train_dir}
    train_cmd="python3 train.py \
	${parameters} \
	--model=NeXtVLADModel \
	--num_readers=8 \
	--learning_rate_decay_examples 2000000 \
        --video_level_classifier_model=LogisticModel \
	--label_loss=CrossEntropyLoss \
	--start_new_model=False \
        --train_data_pattern=${train_data}/train*.tfrecord \
	--train_dir=${train_dir} \
	--frame_features=True \
        --feature_names=rgb,audio \
	--feature_sizes=1024,128 \
	--num_epochs=1
	--batch_size=80 \
	--base_learning_rate=0.0002 \
        --learning_rate_decay=0.8 \
	--l2_penalty=1e-5 \
	--max_step=700000 \
	--num_gpu=2"
    ${train_cmd} > ${dir}/${log_file} 2>&1 &
    pid=$!
    wait $pid
    rm -rf ${train_dir}
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

echo "Benchmark for nextvlad"
gpu_id=`echo $CUDA_VISIBLE_DEVICES | cut -c1`
nvidia-smi --query-compute-apps=used_memory --format=csv -lms 10 > gpu_use.log 2>&1 &
nvidia_pid=$!
train
kill $nvidia_pid
awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "Max=", max}' gpu_use.log

