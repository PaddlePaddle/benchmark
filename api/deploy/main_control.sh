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
feature=("speed" "accuracy")

for line in `cat $auto_file`
do
    api_name=$(echo $line| cut -d',' -f1)
    api=$(echo $line| cut -d',' -f2)
    json_file=$(echo $line| cut -d',' -f3)
    backward=$(echo $line| cut -d',' -f4)
    
    if [ "$json_file" != "None" ]
    then
        casesNum=$(grep "op" $path/$json_dir/$json_file |wc -l)
    else
        casesNum=1
    fi

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
                        repeat=1000
                    else
                        repeat=100
                    fi
                    if [ "$json_file" != "None" ]
                    then 
                        jfile_name=$path/$json_dir/${json_file}
                    else
                        jfile_name=None
                    fi
                    logfile=$res_dir/${api_name}"-"${framwork[$k]}"_"${device[$j]}"_"${feature[$n]}"_"${i}".txt"
                    if [ -e $logfile ]
                    then
                        echo "" >  $logfile
                    fi
                    python $path/$py_dir/${api}.py \
                      --api_name ${api_name} \
                      --task ${feature[$n]} \
                      --framework ${framwork[$k]} \
                      --json_file ${jfile_name} \
                      --config_id $i \
                      --backward ${backward} \
                      --use_gpu ${device_set[$j]} \
                      --repeat $repeat \
                      --log_level 0 > $logfile 2>&1
                done
            done
        done
    done
done
