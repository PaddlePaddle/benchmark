DATASET_PATH=${PWD}/data/cityscape/
INIT_WEIGHTS_PATH=${PWD}/deeplabv3plus_xception65_initialize
SAVE_WEIGHTS_PATH=${PWD}/output/model
echo $DATASET_PATH
#export FLAGS_fraction_of_gpu_memory_to_use=0.0
#GLOG_vmodule=inplace_op_pass=3,eager_deletion_pass=10,memory_optimize_pass=3,parallel_executor=10 GLOG_logtostderr=1 python ./train.py  --batch_size=2  --train_crop_size=513  --total_step=50  --init_weights_path=$INIT_WEIGHTS_PATH --save_weights_path=$SAVE_WEIGHTS_PATH  --dataset_path=$DATASET_PATH --parallel=True
# benchmark command:
# sinlge card
if [ $# -ne 1 ]; then
  echo "Usage: "
  echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh speed"
  exit
fi

fun(){
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    gpus=${#arr[*]}
    batch_size=`expr 2 \* $gpus`
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$gpus, batch_size=$batch_size"
    python ./train.py --batch_size=$batch_size --train_crop_size=513 --total_step=80 \
           --init_weights_path=$INIT_WEIGHTS_PATH --save_weights_path=$SAVE_WEIGHTS_PATH  \
           --dataset_path=$DATASET_PATH --parallel=True > log 2>&1
}
if [ $1 = 'mem' ]
then
    echo "test for $1"
    export FLAGS_fraction_of_gpu_memory_to_use=0.001
    gpu_id=`echo $CUDA_VISIBLE_DEVICES |cut -c1`
    nvidia-smi --id=$gpu_id --query-compute-apps=used_memory --format=csv -lms 100 > gpu_use.log 2>&1 &
    gpu_memory_pid=$!
    fun
    kill $gpu_memory_pid
    awk 'BEGIN {max=0} {if(NR>1){if ($1 > max) max=$1}} END {print "Max_mem_used=", max}' gpu_use.log
else
    echo "test for $1"
    fun
    awk 'BEGIN{count=0} {if(NF==6)count+=1;{if(count>30){res_c+=1;res_time+=$6;}}}END{print "all_step:",count,"\tavg_time:",(res_time/res_c)}' log
fi
# python ./train.py  --batch_size=8  --train_crop_size=513  --total_step=50  --init_weights_path=$INIT_WEIGHTS_PATH --save_weights_path=$SAVE_WEIGHTS_PATH  --dataset_path=$DATASET_PATH --parallel=True
# 6 cards
# python ./train.py  --batch_size=12  --train_crop_size=513  --total_step=50  --init_weights_path=$INIT_WEIGHTS_PATH --save_weights_path=$SAVE_WEIGHTS_PATH  --dataset_path=$DATASET_PATH --parallel=True
# 8 cards
#python ./train.py  --batch_size=16  --train_crop_size=513  --total_step=50  --init_weights_path=$INIT_WEIGHTS_PATH --save_weights_path=$SAVE_WEIGHTS_PATH  --dataset_path=$DATASET_PATH --parallel=True
#python ./train.py  --batch_size=57  --train_crop_size=513  --total_step=50  --init_weights_path=$INIT_WEIGHTS_PATH --save_weights_path=$SAVE_WEIGHTS_PATH  --dataset_path=$DATASET_PATH --parallel=True




#max batch size test
#GLOG_vmodule=legacy_allocator=3 python ./train.py  --batch_size=7  --train_crop_size=513  --total_step=50  --init_weights_path=$INIT_WEIGHTS_PATH --save_weights_path=$SAVE_WEIGHTS_PATH  --dataset_path=$DATASET_PATH --parallel=True
#GLOG_vmodule=legacy_allocator=3 python ./train.py  --batch_size=8  --train_crop_size=513  --total_step=50  --init_weights_path=$INIT_WEIGHTS_PATH --save_weights_path=$SAVE_WEIGHTS_PATH  --dataset_path=$DATASET_PATH --parallel=True
