#!/bin/bash
# usage
# bash benchmark/run_benchmark.sh sp
# bash benchmark/run_benchmark.sh mp  

model_name="tensorflow_east_resnet50"

mode=$1

batch_list=(8 16)

for batch in ${batch_list[@]}; do

    if [ ${mode} = "sp" ]; then
        export CUDA_VISIBLE_DEVICES=0
        train_cmd="python3.7 multigpu_train.py --gpu_list="0" --input_size=640 --batch_size_per_gpu=${batch} --checkpoint_path='' --text_scale=512 --training_data_path=./icdar2015/ --geometry=RBOX --learning_rate=0.0001 --num_readers=24 --pretrained_model_path=./resnet_v1_50.ckpt --max_steps=200"
        num_gpu_devices=1
    else
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        train_cmd="python3.7 multigpu_train.py --gpu_list="0,1,2,3" --input_size=640 --batch_size_per_gpu=${batch} --checkpoint_path='' --text_scale=512 --training_data_path=./icdar2015/ --geometry=RBOX --learning_rate=0.0001 --num_readers=24 --pretrained_model_path=./resnet_v1_50.ckpt --max_steps=100"
        num_gpu_devices=8
    fi
 
    echo $train_cmd
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}   
    fp_item="fp32"
    log_file="${run_log_path}/${model_name}_${mode}_bs${batch}_${fp_item}_${num_gpu_devices}"
    echo $log_file

    timeout 15m ${train_cmd} > ${log_file} 2>&1
    #eval "${train_cmd} > ${log_file} 2>&1"
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
        export job_fail_flag=1
    else
        echo -e "${model_name}, SUCCESS"
        export job_fail_flag=0
    fi
    kill -9 `ps -ef|grep 'python3.7'|awk '{print $2}'`

    if [ $mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
    analysis_cmd="python3.7 benchmark/analysis.py --filename ${log_file}  --mission_name ${model_name} --run_mode ${mode} --direction_id 0 --keyword 'examples/second' --base_batch_size ${batch} --skip_steps 1 --gpu_num ${num_gpu_devices} --index 1  --model_mode=-1  --ips_unit='examples/second'  --position=10"
    eval $analysis_cmd
done
