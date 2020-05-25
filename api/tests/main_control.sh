#! /bin/bash

export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

jsonapi_dir="examples2"
path=$(cd $(dirname $0);pwd)
files=$(ls $path/$jsonapi_dir)

res_dir="result"

if [ ! -d $res_dir ]
then
    mkdir $res_dir
fi

paddle_cpu_perf_suffix="_paddle_cpu_perf.txt"
paddle_cpu_accuracy_suffix="_paddle_cpu_accuracy.txt"
paddle_gpu_perf_suffix="_paddle_gpu_perf.txt"
paddle_gpu_accuracy_suffix="_paddle_gpu_accuracy.txt"
tf_cpu_perf_suffix="_tf_cpu_perf.txt"
tf_cpu_accuracy_suffix="_tf_cpu_accuracy.txt"
tf_gpu_perf_suffix="_tf_gpu_perf.txt"
tf_gpu_accuracy_suffix="_tf_gpu_accuracy.txt"


for file in $files
do
    casesNum=$(grep "op" $path/$jsonapi_dir/$file |wc -l)
    filename=$(basename $file .json)
    for((i=0;i<casesNum;i++));
    do
        paddle_cpu_perf_logfile=$res_dir/${filename}"-"$i$paddle_cpu_perf_suffix
        if [ -e $paddle_cpu_perf_logfile ]
        then
            echo "" >  $paddle_cpu_perf_logfile
        fi
        python ${filename}.py \
          --framework "paddle" \
          --json_file $path/$jsonapi_dir/${file} \
          --config_id $i \
          --check_output False \
          --profiler "none" \
          --backward False \
          --use_gpu False \
          --repeat 100 \
          --log_level 0 > $paddle_cpu_perf_logfile
    done
done


for file in $files
do
    ccasesNum=$(grep "op" $path/$jsonapi_dir/$file |wc -l)
    filename=$(basename $file .json)
    for((i=0;i<casesNum;i++));
    do
        paddle_gpu_perf_logfile=$res_dir/${filename}"-"$i$paddle_gpu_perf_suffix
        if [ -e $paddle_gpu_perf_logfile ]
        then
            echo "" >  $paddle_gpu_perf_logfile
        fi
        python ${filename}.py \
          --framework "paddle" \
          --json_file $path/$jsonapi_dir/${file} \
          --config_id $i \
          --check_output False \
          --profiler "none" \
          --backward False \
          --use_gpu True \
          --repeat 10000 \
          --log_level 0 > $paddle_gpu_perf_logfile
    done
done



for file in $files
do
    casesNum=$(grep "op" $path/$jsonapi_dir/$file |wc -l)
    filename=$(basename $file .json)
    for((i=0;i<casesNum;i++));
    do
        tf_cpu_perf_logfile=$res_dir/${filename}"-"$i$tf_cpu_perf_suffix
        if [ -e $tf_cpu_perf_logfile ]
        then
            echo "" >  $tf_cpu_perf_logfile
        fi
        python ${filename}.py \
          --framework "tensorflow" \
          --json_file $path/$jsonapi_dir/${file} \
          --config_id $i \
          --check_output False \
          --profiler "none" \
          --backward False \
          --use_gpu False \
          --repeat 100 \
          --log_level 0 > $tf_cpu_perf_logfile
    done
done



for file in $files
do
    casesNum=$(grep "op" $path/$jsonapi_dir/$file |wc -l)
    filename=$(basename $file .json)
    for((i=0;i<casesNum;i++));
    do
        tf_gpu_perf_logfile=$res_dir/${filename}"-"$i$tf_gpu_perf_suffix
        if [ -e $tf_gpu_perf_logfile ]
        then
            echo "" >  $tf_gpu_perf_logfile
        fi
        python ${filename}.py \
          --framework "tensorflow" \
          --json_file $path/$jsonapi_dir/${file} \
          --config_id $i \
          --check_output False \
          --profiler "none" \
          --backward False \
          --use_gpu True \
          --repeat 10000 \
          --log_level 0 > $tf_gpu_perf_logfile
    done
done




for file in $files
do
    casesNum=$(grep "op" $path/$jsonapi_dir/$file |wc -l)
    filename=$(basename $file .json)
    for((i=0;i<casesNum;i++));
    do
        paddle_cpu_accuarcy_logfile=$res_dir/${filename}"-"$i$paddle_cpu_accuracy_suffix
        if [ -e $paddle_cpu_accuarcy_logfile ]
        then
            echo "" >  $paddle_cpu_accuarcy_logfile
        fi
        python ${filename}.py \
          --task "accuracy" \
          --framework "paddle" \
          --json_file $path/$jsonapi_dir/${file} \
          --config_id $i \
          --check_output False \
          --profiler "none" \
          --backward False \
          --use_gpu False \
          --repeat 1 \
          --log_level 0 > $paddle_cpu_accuarcy_logfile
    done
done




for file in $files
do
    casesNum=$(grep "op" $path/$jsonapi_dir/$file |wc -l)
    filename=$(basename $file .json)
    for((i=0;i<casesNum;i++));
    do
        paddle_gpu_accuarcy_logfile=$res_dir/${filename}"-"$i$paddle_gpu_accuracy_suffix
        if [ -e $paddle_gpu_accuarcy_logfile ]
        then
            echo "" >  $paddle_gpu_accuarcy_logfile
        fi
        python ${filename}.py \
          --task "accuracy" \
          --framework "paddle" \
          --json_file $path/$jsonapi_dir/${file} \
          --config_id $i \
          --check_output False \
          --profiler "none" \
          --backward False \
          --use_gpu True \
          --repeat 1 \
          --log_level 0 > $paddle_gpu_accuarcy_logfile
    done
done
