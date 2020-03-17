#!/bin/bash

dir=$1
rm $dir/final_res.txt

for i in $dir/RES/*.res
do
  echo "------------------$i" >> $dir/final_res.txt
  cat $i | grep "FINAL_RESULT" >> $dir/final_res.txt
  cat $i | grep "^MAX_GPU_MEMORY_USE" >> $dir/final_res.txt 
  cat $i | grep "^AVG_CPU_USE" >> $dir/final_res.txt
  cat $i | grep "^AVG_GPU_USE" >> $dir/final_res.txt
  echo "                  " >> $dir/final_res.txt
  echo "                  " >> $dir/final_res.txt
done
