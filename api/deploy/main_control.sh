#! /bin/bash

export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

AUTO_RUN_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}/../")" && pwd )"
export PYTHONPATH=${AUTO_RUN_ROOT}:${PYTHONPATH}

python collect_api_info.py --info_file  api_info.txt --support_api_file support_api_list.txt

py_dir="../tests"
api_info_file="api_info.txt"
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
no_backward=()

for line in `cat $api_info_file`
do
    api_num=$[$api_num+1]
    api_name=$(echo $line| cut -d',' -f1)
    api=$(echo $line| cut -d',' -f2)
    json_file=$(echo $line| cut -d',' -f3)
    backward=$(echo $line| cut -d',' -f4)

    if [ ${backward} = False ]; then  
        length=${#no_backward[@]}
        length=$[$length+1]
        no_backward[${length}]=${api_name}  
    fi

    if [ "$json_file" != "None" ]
    then
        cases_num=$(grep ""op"" $path/$json_dir/$json_file |wc -l)
        json_file_name=$path/$json_dir/${json_file}
    else
        cases_num=1
        json_file_name=None
    fi

    for((i=0;i<cases_num;i++));
    do
        for j in "${!device[@]}";
        do 
            for k in "${!framwork[@]}";
            do 
                for n in "${!feature[@]}";
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
                        logfile=$res_dir/${api_name}"-"${framwork[$k]}"_"${device[$j]}"_"${feature[$n]}"_""${direction[$m]}""_"${i}".txt"
                        echo "api_name: "${api_name}", api: "${api}", "${direction[$m]}", json_file: "${json_file}", json_id: "${i}", "${logfile}

                        python -m tests.launch ${AUTO_RUN_ROOT}/deploy/${api}.py \
                          --api_name ${api_name} \
                          --task ${feature[$n]} \
                          --framework ${framwork[$k]} \
                          --json_file ${json_file_name} \
                          --config_id $i \
                          --backward ${direction_set[$m]} \
                          --use_gpu ${device_set[$j]} \
                          --repeat $repeat \
                          --log_level 0 > $logfile 2>&1
                   done
                done
            done
       done
    done
    echo -e "\n"
done
echo -e "\nSUMMARY: The number of whole API: "${api_num}
echo -e "\nSUMMARY: APIs which has no backwards: "${no_backward[@]}
