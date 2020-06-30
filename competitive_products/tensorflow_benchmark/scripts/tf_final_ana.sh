#!/bin/bash

dir=$1
rm $dir/final_data.txt

for i in $dir/RES/1_*
do
  echo "------------------$i" >> $dir/final_data.txt 
  cat $i | grep "Avg:" >> $dir/final_data.txt 
  #awk '/AVG/{i++}i==2' >> $dir/final_data.txt
  cat $i | grep "FPS:" >> $dir/final_data.txt 
  cat $i | grep "FINAL_RESULT" >> $dir/final_data.txt 
  cat $i | grep "^AVG_CPU_USE" >> $dir/final_data.txt 
  cat $i | grep "^AVG_GPU_USE" >> $dir/final_data.txt 
done

echo "                " >> $dir/final_data.txt 
echo "                " >> $dir/final_data.txt 
echo "                " >> $dir/final_data.txt 

for j in $dir/RES/2_*
do
  echo "------------------$j" >> $dir/final_data.txt 
  cat $j | grep "^MAX_GPU_MEMORY_USE" >> $dir/final_data.txt 
done

