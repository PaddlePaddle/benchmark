DATASET_PATH=${PWD}/data/cityscape/
INIT_WEIGHTS_PATH=${PWD}/deeplabv3plus_xception65_initialize
SAVE_WEIGHTS_PATH=${PWD}/output/model
echo $DATASET_PATH
#export FLAGS_fraction_of_gpu_memory_to_use=0.0
#GLOG_vmodule=inplace_op_pass=3,eager_deletion_pass=10,memory_optimize_pass=3,parallel_executor=10 GLOG_logtostderr=1 python ./train.py  --batch_size=2  --train_crop_size=513  --total_step=50  --init_weights_path=$INIT_WEIGHTS_PATH --save_weights_path=$SAVE_WEIGHTS_PATH  --dataset_path=$DATASET_PATH --parallel=True
# benchmark command:
# sinlge card
python ./train.py  --batch_size=2  --train_crop_size=513  --total_step=50  --init_weights_path=$INIT_WEIGHTS_PATH --save_weights_path=$SAVE_WEIGHTS_PATH  --dataset_path=$DATASET_PATH --parallel=True
# 4 cards
# python ./train.py  --batch_size=8  --train_crop_size=513  --total_step=50  --init_weights_path=$INIT_WEIGHTS_PATH --save_weights_path=$SAVE_WEIGHTS_PATH  --dataset_path=$DATASET_PATH --parallel=True
# 6 cards
# python ./train.py  --batch_size=12  --train_crop_size=513  --total_step=50  --init_weights_path=$INIT_WEIGHTS_PATH --save_weights_path=$SAVE_WEIGHTS_PATH  --dataset_path=$DATASET_PATH --parallel=True
# 8 cards
#python ./train.py  --batch_size=16  --train_crop_size=513  --total_step=50  --init_weights_path=$INIT_WEIGHTS_PATH --save_weights_path=$SAVE_WEIGHTS_PATH  --dataset_path=$DATASET_PATH --parallel=True
#python ./train.py  --batch_size=57  --train_crop_size=513  --total_step=50  --init_weights_path=$INIT_WEIGHTS_PATH --save_weights_path=$SAVE_WEIGHTS_PATH  --dataset_path=$DATASET_PATH --parallel=True




#max batch size test
#GLOG_vmodule=legacy_allocator=3 python ./train.py  --batch_size=7  --train_crop_size=513  --total_step=50  --init_weights_path=$INIT_WEIGHTS_PATH --save_weights_path=$SAVE_WEIGHTS_PATH  --dataset_path=$DATASET_PATH --parallel=True
#GLOG_vmodule=legacy_allocator=3 python ./train.py  --batch_size=8  --train_crop_size=513  --total_step=50  --init_weights_path=$INIT_WEIGHTS_PATH --save_weights_path=$SAVE_WEIGHTS_PATH  --dataset_path=$DATASET_PATH --parallel=True
