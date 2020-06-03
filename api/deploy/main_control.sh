#! /bin/bash

export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

path=$(cd $(dirname $0);pwd)
test_dir="../tests"
export PYTHONPATH=${path}/${test_dir}:${PYTHONPATH}

api_info_file=$path/api_info.txt
python ${path}/collect_api_info.py --info_file  ${api_info_file} --support_api_file ${path}/support_api_list.txt

json_dir=$1
res_dir=$2
if [ ! -d $res_dir ]
then
    mkdir $res_dir
fi

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
        json_file_name=${PWD}/$json_dir/${json_file}
        cases_num=$(grep '"op"' ${json_file_name} |wc -l)
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
                        if [ ${framwork[$k]} = "tensorflow" ] && [ ${feature[$n]} = "accuracy" ]; then
                            continue
                        fi
                        if [ "${device[$j]}" = "gpu" ]; then
                            repeat=1000
                        else
                            repeat=100
                        fi
                        logfile=$res_dir/${api_name}"-"${framwork[$k]}"_"${device[$j]}"_"${feature[$n]}"_""${direction[$m]}""_"${i}".txt"
                        echo "api_name: "${api_name}", api: "${api}", "${direction[$m]}", json_file: "${json_file}", json_id: "${i}", "${logfile}

                        python -m launch ${path}/${test_dir}/${api}.py \
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
