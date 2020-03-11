#!/bin/bash
set -x

echo "CUDA_VISIBLE_DEVICES=7 bash run_transformer.sh speed|mem big|base /log/path"

index=$1
mode=$2
model_name=transformer_$mode
run_log_path=${3:-$(pwd)}

device=${CUDA_VISIBLE_DEVICES//,/ }
arr=($device)
num_gpu_devices=${#arr[*]}

base_batch_size=4096
batch_size=`expr ${base_batch_size} \* $num_gpu_devices`
log_file=${run_log_path}/log_${index}_${model_name}_${num_gpu_devices}

export HOME=/home/workspace/t2t_data
PROBLEM=translate_ende_wmt_bpe32k
MODEL=transformer
HPARAMS=${model_name}

DATA_DIR=$HOME/t2t_data_ende16_bpe/
TMP_DIR=$HOME/t2t_datagen_ende16_bpe/
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS

#mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR
mkdir -p $TRAIN_DIR

function _train(){

if [ $index = "speed" ]; then
    sed -i '81c \  config.gpu_options.allow_growth = False' /usr/local/lib/python2.7/dist-packages/tensor2tensor/utils/trainer_lib.py
elif [ $index = "mem" ]; then
    echo "this index is: "$index
    sed -i '81c \  config.gpu_options.allow_growth = True' /usr/local/lib/python2.7/dist-packages/tensor2tensor/utils/trainer_lib.py
fi

rm -rf ${HOME}/*
cpu_num=$(cat /proc/cpuinfo | grep processor | wc -l)
gpu_num=$(nvidia-smi -L|wc -l)
gpu_id=`echo $CUDA_VISIBLE_DEVICES | cut -c1`
nvidia-smi --id=$gpu_id --query-compute-apps=used_memory --format=csv -lms 100 > ${log_file}_gpu_use.log 2>&1 &            #for mem max
gpu_memory_pid=$!


job_bt=`date '+%Y%m%d%H%M%S'`
#data generate
t2t-datagen \
   --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM > ${log_file} 2>&1

# Train


# *  If you run out of memory, add --hparams='batch_size=1024'.
t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --keep_checkpoint_max=0 \
  --local_eval_frequency=10000 \
  --eval_steps=500 \
  --train_steps=1000 \
  --eval_throttle_seconds=8640000 \
  --train_steps=1000000 \
  --worker_gpu=${num_gpu_devices} \
  --hparams="batch_size=${base_batch_size}" >> ${log_file} 2>&1 &
train_pid=$!

if [ ${num_gpu_devices} = 1 ]; then
    sleep 800
elif [ ${num_gpu_devices} = 8 ]; then
    sleep 1100
else
    sleep 900
fi
kill -9 ${train_pid}

job_et=`date '+%Y%m%d%H%M%S'`
hostname=`echo $(hostname)|awk -F '.baidu.com' '{print $1}'`
monquery -n $hostname -i GPU_AVERAGE_UTILIZATION -s $job_bt -e $job_et -d 60 > ${log_file}_gpu_avg_utilization
monquery -n $hostname -i CPU_USER -s $job_bt -e $job_et -d 60 > ${log_file}_cpu_use
awk '{if(NR>1 && $3 >0){time+=$3;count+=1}} END{if(count>0) avg=time/count; else avg=0; printf("AVG_GPU_USE=%.2f\n" ,avg*'${gpu_num}')}' ${log_file}_gpu_avg_utilization
awk '{if(NR>1 && $3 >0){time+=$3;count+=1}} END{if(count>0) avg=time/count; else avg=0; printf("AVG_CPU_USE=%.2f\n" ,avg*'${cpu_num}')}' ${log_file}_cpu_use
kill ${gpu_memory_pid}
killall -9 nvidia-smi
cat ${log_file}_gpu_use.log | tr -d ' MiB' | awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "MAX_GPU_MEMORY_USE=", max}'
}
_train
cat ${log_file} |  grep "INFO:tensorflow:global_step/sec:" | awk -F " " '{print$2}'  | awk '{sum+=$1} END {print "FINAL_RESULT = ", sum/NR}' 
