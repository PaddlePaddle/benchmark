### 模型库的 case 调度脚本
# example: bash trian_benchmark.sh models/PP-LCNet_x1_0/dynamic_bs16_fp32_DP_N1C1.sh
function _train(){
    echo --------- running $1
    if [[ $1 =~ "N1C1" ]];then
        export CUDA_VISIBLE_DEVICES=0;
    else
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;
    fi
    source $1
}
function _analysis_log(){
    # 解析版本信息
    str_tmp=$(echo `pip list|grep paddlepaddle-gpu|awk -F ' ' '{print $2}'`)
    export frame_version=${str_tmp%%.post*}
    export frame_commit=$(python -c "import paddle; print(paddle.version.commit)" | tail -n 1)
    export model_branch=`git symbolic-ref HEAD 2>/dev/null | cut -d"/" -f 3`
    export model_commit=$(git log|head -n1|awk '{print $2}')
    echo "---------frame_version is ${frame_version}"
    echo "---------Paddle commit is ${frame_commit}"
    echo "---------Model commit is ${model_commit}"
    echo "---------model_branch is ${model_branch}"
    
    # speed_log_file 信息修改 analysis.py ，通过 log_file来获取
    if [ "${separator}" == "" ]; then
        separator="None"
    fi
    analysis_options=""
    if [ "${position}" != "" ]; then
        analysis_options="${analysis_options} --position ${position}"
    fi
    if [ "${range}" != "" ]; then
        analysis_options="${analysis_options} --range ${range}"
    fi
    if [ "${model_mode}" != "" ]; then
        analysis_options="${analysis_options} --model_mode ${model_mode}"
    fi
    
    cmd="python ${BENCHMARK_ROOT}/scripts/analysis.py \
            --filename ${log_file} \
            --speed_log_file ${speed_log_file} \
            --model_name ${model_name} \
            --base_batch_size ${batch_size} \
            --run_mode ${run_mode} \
            --fp_item ${fp_item} \
            --keyword ${keyword} \
            --skip_steps ${skip_steps} \
            --device_num ${device_num} \
            --speed_unit ${speed_unit} \
            --convergence_key ${convergence_key} \
            --is_large_model ${is_large_model:-"False"} \
            --separator "${separator}" ${analysis_options}"
    echo $cmd
    eval $cmd
}
# case 执行
job_bt=`date '+%Y%m%d%H%M%S'`
_train $1
job_et=`date '+%Y%m%d%H%M%S'`
export model_run_time=$((${job_et}-${job_bt}))
_analysis_log $1

# python /paddle/pdc/benchmark-frame/tools/scripts/analysis.py --filename /paddle/paddlex/PaddleX/PaddleX_PaddleClas_PP-HGNet_small_bs128_fp32_DP_N1C1_log --speed_log_file /paddle/paddlex/PaddleX/PaddleX_PaddleClas_PP-HGNet_small_bs128_fp32_DP_N1C1_speed --model_name PP-HGNet_small_bs128_fp32_DP --base_batch_size --run_mode --fp_item --keyword ips: --skip_steps 10 --device_num --speed_unit samples/s --convergence_key loss: --is_large_model False --separator None