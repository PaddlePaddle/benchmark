#! /bin/bash
dir_path=/benchmark/op_GPU_logs/1016/Paddle/1.8
rm -rf  $dir_path/1.8_op_GPU.log
rm -rf $dir_path/*tensorflow*
rm -rf $dir_path/*accuracy*

rm -rf $dir_path/tmp
if [ ! -d $dir_path/tmp ];then
mkdir $dir_path/tmp
fi

declare -A dict
dict=([conv2d_transpose]="conv_transpose2d" [elementwise_add]="add" [elementwise_div]="divide" [elementwise_max]="maximum" [elementwise_min]="minimum" [elementwise_mul]="multiply" [elementwise_pow]="pow" [fc]="linear" [fill_constant]="fill" [l2_normalize]="normalize" [reduce_mean]="mean" [reduce_prod]="prod" [reduce_sum]="sum" [resize_bilinear]="interp_bilinear" [resize_nearest]="interp_nearest" [expand]="tile")

file_path=$(ls $dir_path)
for filename in $file_path
do
   echo $dir_path/$filename
   
   op_list='conv2d_transpose elementwise_add elementwise_div elementwise_max elementwise_min elementwise_mul elementwise_pow fc fill_constant l2_normalize reduce_mean reduce_prod reduce_sum resize_bilinear resize_nearest expand'
   for op_name in $op_list
   do
       if [[ $filename =~ ^${op_name}.* ]]; then
        filename_1=`echo $filename| sed s/$op_name/${dict[${op_name}]}/g`
        cp $dir_path/$filename  $dir_path/tmp/$filename_1
        rm -rf $dir_path/$filename
       fi 
   done
   # pool2d
   if [[ $filename =~ ^pool2d.* ]]; then
        tmp=${filename%-*}
        config_id=`echo ${tmp#*_}`
        echo $config_id                 
        if [ $config_id == 2 ] || [ $config_id == 3 ] ;then
           filename_1=`echo $filename| sed s/pool2d/max_pool2d/g`
           cp $dir_path/$filename  $dir_path/tmp/$filename_1
           rm -rf $dir_path/$filename
        else
           filename_1=`echo $filename| sed s/pool2d/avg_pool2d/g`
           cp $dir_path/$filename  $dir_path/tmp/$filename_1
           rm -rf $dir_path/$filename
        fi  
   fi
done

cp -r $dir_path/tmp/*  $dir_path/
# rm -rf $dir_path/tmp
