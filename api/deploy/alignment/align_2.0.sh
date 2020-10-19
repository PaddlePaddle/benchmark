#! /bin/bash
dir_path=/benchmark/op_GPU_logs/1016/Paddle/2.0
rm -rf  $dir_path/2.0_op_GPU.log
rm -rf $dir_path/*tensorflow*
rm -rf $dir_path/*accuracy*

file_path=$(ls $dir_path)
rm -rf $dir_path/tmp
if [ ! -d $dir_path/tmp ];then
mkdir -p $dir_path/tmp
fi

for filename in $file_path
do
echo $dir_path/$filename

# config num, conv2d conv2d_transpose softmax depthwise_conv2d
op_list='conv2d conv_transpose2d softmax depthwise_conv2d'
    
tmp=${filename%-*}
# op_name
echo $tmp
op_name=${tmp%_*}
echo $op_name
# config_id
config_id=`echo $tmp | awk -F '_' '{print $NF}'`

for op in $op_list
do
if [[ $op_name == $op ]]; then
    echo $op_name
    tmp=${filename%-*}    
    config_id=`echo $tmp | awk -F '_' '{print $NF}'` 
    echo $config_id

    left=${filename%-*}
    left=${left%_*}'_'  
    right='-'${filename#*-}
 
    let config_id_1=config_id*2
    echo $config_id_1
    filename_1=${left}${config_id_1}${right}
    cp $dir_path/$filename  $dir_path/tmp/$filename_1
    
    let config_id_2=config_id_1+1
    echo $config_id_2
    filename_2=${left}${config_id_2}${right}
    cp $dir_path/tmp/$filename_1 $dir_path/tmp/$filename_2 
    echo $dir_path/tmp/$filename_2
    rm -rf $dir_path/$filename
fi
done

# avg_pool2d max_pool2d
declare -A dict
dict=([0]="0" [1]="4" [2]="6")

if [[ $op_name == 'max_pool2d' ]]; then
    if [ $config_id == 0 ];then
       left=${filename%-*}
       left=${left%_*}'_'
       right='-'${filename#*-}
       
       config_id_1=2
       filename_1=${left}${config_id_1}${right}
       cp $dir_path/$filename  $dir_path/tmp/$filename_1 
    
       config_id_2=3
       filename_2=${left}${config_id_2}${right}
       cp $dir_path/tmp/$filename_1 $dir_path/tmp/$filename_2
       rm -rf $dir_path/$filename 
       fi    
fi
# avg_pool2d
if [[ $op_name == 'avg_pool2d' ]]; then
       left=${filename%-*}
       left=${left%_*}'_'
       right='-'${filename#*-}
       echo config_id
       config_id_1=${dict[${config_id}]}
       echo config_id_1
       filename_1=${left}${config_id_1}${right}
       cp $dir_path/$filename  $dir_path/tmp/$filename_1

       let config_id_2=config_id_1+1
        echo config_id_2
       filename_2=${left}${config_id_2}${right}
       cp $dir_path/tmp/$filename_1 $dir_path/tmp/$filename_2
       rm -rf $dir_path/$filename
fi
done

cp $dir_path/tmp/*  $dir_path/
# rm -rf $dir_path/tmp

