#!bin/bash
set -xe

if [[ $# -lt 3 ]]; then
    echo "running job dict is {1: speed, 2:mem, 3:profiler, 6:max_batch_size}"
    echo "Usage: "
    echo "  CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1|2|3|6 base|big sp|mp 1000(max_iter)"
    exit
fi

function _set_params(){
    index="$1"
    model_type="$2"
    run_mode=${3:-"sp"}
    max_iter=${4}
    if [[ ${index} -eq 3 ]]; then is_profiler=1; else is_profiler=0; fi
   
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
    profiler_path=${PROFILER_LOG_DIR:-$(pwd)}
 
    model_name="transformer_"${model_type}
    skip_steps=3
    keyword="speed:"
    separator=" "
    position=-2
    model_mode=1

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=($device)
    num_gpu_devices=${#arr[*]}

    if [[ ${index} -eq 6 ]]; then base_batch_size=12000; else base_batch_size=4096; fi
    batch_size=`expr ${base_batch_size} \* $num_gpu_devices`
    log_file=${run_log_path}/${model_name}_${index}_${num_gpu_devices}_${run_mode}
    log_with_profiler=${profiler_path}/${model_name}_3_${num_gpu_devices}_${run_mode}
    profiler_path=${profiler_path}/profiler_${model_name}
    if [[ ${is_profiler} -eq 1 ]]; then log_file=${log_with_profiler}; fi
    log_parse_file=${log_file}

}

function _set_env(){
    export FLAGS_fraction_of_gpu_memory_to_use=1.0
    export FLAGS_eager_delete_tensor_gb=0.0
    export FLAGS_memory_fraction_of_eager_deletion=0.99999
    export FLAGS_allocator_strategy=naive_best_fit
}

function _train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
    #cd ../../../../models/PaddleNLP/neural_machine_translation/transformer/
    # base model
    if [ ${model_type} = 'big' ]; then
        train_cmd=" --do_train True \
        --epoch 30 \
        --src_vocab_fpath data/vocab.bpe.32000 \
        --trg_vocab_fpath data/vocab.bpe.32000 \
        --special_token <s> <e> <unk> \
	--training_file data/train.tok.clean.bpe.32000.en-de \
	--batch_size ${base_batch_size}
	--n_head 16 \
        --d_model 1024 \
        --d_inner_hid 4096 \
        --is_profiler=${is_profiler} \
        --profiler_path=${profiler_path} \
        --max_iter=${max_iter} \
        --prepostprocess_dropout 0.3"
    else
        train_cmd=" --do_train True \
        --epoch 30 \
        --src_vocab_fpath data/vocab.bpe.32000 \
        --trg_vocab_fpath data/vocab.bpe.32000 \
        --special_token <s> <e> <unk> \
        --is_profiler=${is_profiler} \
        --profiler_path=${profiler_path} \
        --max_iter=${max_iter} \
	--training_file data/train.tok.clean.bpe.32000.en-de \
	--batch_size ${base_batch_size}"
    fi

    case ${run_mode} in
    sp) train_cmd="python -u main.py "${train_cmd} ;;
    mp)
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --selected_gpus=$CUDA_VISIBLE_DEVICES main.py "${train_cmd}
        log_parse_file="mylog/workerlog.0" ;;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac

    ${train_cmd} > ${log_file} 2>&1 
    kill -9 `ps -ef|grep python |awk '{print $2}'`

    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_set_env
_run
