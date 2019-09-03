#!bin/bash
set -xe

if [[ $# -lt 3 ]]; then
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run.sh speed|mem|maxbs base|big sp|mp /ssd1/ljh/logs"
    exit
fi

function _set_params(){
    index="$1"
    model_type="$2"
    run_mode=${3:-"sp"}
    run_log_path=${4:-$(pwd)}

    model_name="transformer_"${model_type}
    skip_steps=3
    keyword="loss_rpn_bbox"
    separator='" "'
    position=-2
    model_mode=1

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    if [[ ${index} = "maxbs" ]]; then base_batch_size=12000; else base_batch_size=4096; fi
    batch_size=`expr ${base_batch_size} \* $num_gpu_devices`
    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_parse_file=${log_file}

}

function _set_env(){
    export FLAGS_fraction_of_gpu_memory_to_use=1.0
    export FLAGS_eager_delete_tensor_gb=0.0
    export FLAGS_memory_fraction_of_eager_deletion=0.99999
}

function _train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
    #cd ../../../../models/PaddleNLP/neural_machine_translation/transformer/
    # base model
    if [ ${model_type} = 'big' ]; then
        train_cmd=" --src_vocab_fpath data/vocab.bpe.32000 \
            --trg_vocab_fpath data/vocab.bpe.32000 \
            --special_token <s> <e> <unk> \
            --train_file_pattern data/train.tok.clean.bpe.32000.en-de \
            --use_token_batch True \
            --batch_size ${base_batch_size} \
            --use_token_batch True \
            --sort_type pool \
            --pool_size 200000 \
            --shuffle True \
            --shuffle_batch True \
            --use_py_reader True \
            --use_mem_opt True \
            --enable_ce False \
            --fetch_steps 100 \
            learning_rate 2.0 \
            warmup_steps 8000 \
            beta2 0.997 \
            d_model 1024 \
            d_inner_hid 4096 \
            n_head 16 \
            prepostprocess_dropout 0.3 \
            attention_dropout 0.1 \
            relu_dropout 0.1 \
            weight_sharing True \
            pass_num 100 \
            max_length 256"
    else
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
    fi

    case ${run_mode} in
    sp) train_cmd="python -u train.py "${train_cmd} ;;
    mp)
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --selected_gpus=$CUDA_VISIBLE_DEVICES train.py "${train_cmd}
        log_parse_file="mylog/workerlog.0" ;;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac

    ${train_cmd} > ${log_file} 2>&1 &
    train_pid=$!
    sleep 600
    kill -9 `ps -ef|grep python |awk '{print $2}'`

    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}

function _run(){
    if [[ ${index} = "mem" ]]; then
        #若测试最大batchsize，FLAGS_fraction_of_gpu_memory_to_use=1
        export FLAGS_fraction_of_gpu_memory_to_use=0.001
        gpu_id=`echo $CUDA_VISIBLE_DEVICES | cut -c1`
        nvidia-smi --id=$gpu_id --query-compute-apps=used_memory --format=csv -lms 100 > gpu_use.log 2>&1 &
        gpu_memory_pid=$!
        _train
        kill ${gpu_memory_pid}
        awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "MAX_GPU_MEMORY_USE=", max}' gpu_use.log
    elif [[ ${index} = "speed" ]]; then
        job_bt=`date '+%Y%m%d%H%M%S'`
        _train
        job_et=`date '+%Y%m%d%H%M%S'`
        hostname=`echo $(hostname)|awk -F '.baidu.com' '{print $1}'`
        gpu_log_file=${log_file}_gpu
        cpu_log_file=${log_file}_cpu
        monquery -n ${hostname} -i GPU_AVERAGE_UTILIZATION -s $job_bt -e $job_et -d 60 > ${gpu_log_file}
        monquery -n ${hostname} -i CPU_USER -s $job_bt -e $job_et -d 60 > ${cpu_log_file}
        cpu_num=$(cat /proc/cpuinfo | grep processor | wc -l)
        gpu_num=$(nvidia-smi -L|wc -l)

        awk '{if(NR>1 && $3 >0){time+=$3;count+=1}} END{if(count>0) avg=time/count; else avg=0; printf("AVG_GPU_USE=%.2f\n" ,avg*'${gpu_num}')}' ${gpu_log_file}
        awk '{if(NR>1 && $3 >0){time+=$3;count+=1}} END{if(count>0) avg=time/count; else avg=0; printf("AVG_CPU_USE=%.2f\n" ,avg*'${cpu_num}')}' ${cpu_log_file}
    else
        _train
        error_string="Please shrink FLAGS_fraction_of_gpu_memory_to_use"
        if [ `grep -c "${error_string}" ${log_parse_file}` -eq 0 ]; then
            echo "MAX_BATCH_SIZE=${batch_size}"
        else
            echo "MAX_BATCH_SIZE=0"
        fi
    fi

    python ${BENCHMARK_ROOT}/utils/analysis.py \
            --filename ${log_file} \
            --keyword ${keyword} \
            --separator ${separator} \
            --position ${position} \
            --base_batch_size ${base_batch_size} \
            --skip_steps ${skip_steps} \
            --model_mode ${model_mode} \
            --model_name ${model_name} \
            --run_mode ${run_mode} \
            --index ${index} \
            --gpu_num ${num_gpu_devices}
}

_set_params
_set_env
_run