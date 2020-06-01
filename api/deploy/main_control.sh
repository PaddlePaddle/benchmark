#! /bin/bash

export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

py_dir="../tests"
auto_file=$1
json_dir=$2
res_dir=$3
if [ ! -d $res_dir ]
then
    mkdir $res_dir
fi

path=$(cd $(dirname $0);pwd)

device=("cpu" "gpu")
device_set=(false true)
framwork=("paddle" "tensorflow")
feature=("perf" "accuracy")
feature_set=(false true)

for line in `cat $auto_file`
do
    api=$(echo $line| cut -d',' -f1)
    json_file=$(echo $line| cut -d',' -f2)
    backend=$(echo $line| cut -d',' -f3)
    
    if [ "$json_file" != "None" ]
    then
        casesNum=$(grep "op" $path/$json_dir/$json_file |wc -l)
        echo $casesNum
        for((i=0;i<casesNum;i++));
        do
            for j in "${!device[@]}";
            do 
                for k in "${!framwork[@]}";
                do 
                    for n in "${!feature[@]}";
                    do 
                        if [ "${device[$j]}" = "gpu" ]
                        then
                            repeat=10000
                        else
                            repeat=100
                        fi
                        if [ "${feature[$n]}" = "accuracy" ]
                        then
                            repeat=1
                        fi
                        logfile=$res_dir/$i"_"${framwork[$k]}"_"${device[$j]}"_"${feature[$n]}".txt"
                        if [ -e $logfile ]
                        then
                            echo "" >  $paddle_cpu_perf_logfile
                        fi
                        python $path/$py_dir/${api}.py \
                          --framework ${framwork[$k]} \
                          --json_file $path/$json_dir/${json_file} \
                          --config_id $i \
                          --check_output ${feature_set[$k]} \
                          --profiler "none" \
                          --backward $backend \
                          --use_gpu ${device_set[$i]} \
                          --repeat $repeat \
                          --log_level 0 > $logfile
                    done
                done
            done
        done
    fi
done
