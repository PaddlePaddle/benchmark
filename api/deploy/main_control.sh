#! /bin/bash

export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

python register.py

export PYTHONPATH=/home/gaowei/padding/tf_com/benchmark/api/tests:${PYTHONPATH}
py_dir="../tests"
api_info_file="auto_run_info.txt"
json_dir=$1
res_dir=$2
if [ ! -d $res_dir ]
then
    mkdir $res_dir
fi

path=$(cd $(dirname $0);pwd)

device=("cpu" "gpu")
device_set=(false true)
direction=("forward" "backward")
direction_set=(false true)
framwork=("paddle" "tensorflow")
feature=("speed" "accuracy")
api_num=0
no_backwards=()

for line in `cat $api_info_file`
do
    api_num=$[$api_num+1]
    api_name=$(echo $line| cut -d',' -f1)
    api=$(echo $line| cut -d',' -f2)
    json_file=$(echo $line| cut -d',' -f3)
    backward=$(echo $line| cut -d',' -f4)
    if [ ${backward} = False ]; then  
        length=${#no_backwards[@]}
        length=$[$length+1]
        no_backwards[${length}]=${api_name}  
    fi
    if [ "$json_file" != "None" ]
    then
        casesNum=$(grep "op" $path/$json_dir/$json_file |wc -l)
    else
        casesNum=1
    fi

    if [ "$json_file" != "None" ]; then
        json_file_name=$path/$json_dir/${json_file}
    else
        json_file_name=None
    fi

    for((i=0;i<casesNum;i++));
    do
        for j in "${!device[@]}";
        do 
            for k in "${!framwork[@]}";
            do 
                for m in "${!direction[@]}"; 
                do
                    if [ ${direction[$m]} = "backward" ] && [ ${backward} = False ]; then
                        continue
                    fi
                    if [ "${device[$j]}" = "gpu" ]; then
                        repeat=1
                    else
                        repeat=1
                    fi
                    logfile=$res_dir/${api_name}"-"${framwork[$k]}"_"${device[$j]}"_speed_""${direction[$m]}""_"${i}".txt"
                    echo "api_name: "${api_name}", api: "${api}", "${direction[$m]}", json_file: "${json_file}", json_id: "${i}
                    echo ${logfile}

                    python -m launch $path/$py_dir/${api}.py \
                      --api_name ${api_name} \
                      --framework ${framwork[$k]} \
                      --json_file ${json_file_name} \
                      --config_id $i \
                      --backward ${direction_set[$m]} \
                      --use_gpu ${device_set[$j]} \
                      --repeat $repeat \
                      --log_level 0 > $logfile 2>&1
               done
            done

            logfile=$res_dir/${api_name}"-paddle_"${device[$j]}"_accuracy_forward_"${i}".txt"
            echo "api_name: "${api_name}", api: "${api}", forward, json_file: "${json_file}", json_id: "${i}
            echo ${logfile}

            python -m launch $path/$py_dir/${api}.py \
              --api_name ${api_name} \
              --task accuracy \
              --framework paddle \
              --json_file ${json_file_name} \
              --config_id $i \
              --backward ${direction_set[$m]} \
              --use_gpu ${device_set[$j]} \
              --repeat $repeat \
              --log_level 0 > $logfile 2>&1
       done
    done
done
echo "The number of whole API: "${api_num}
echo "APIs which has no backwards: "${no_backwards[@]}
