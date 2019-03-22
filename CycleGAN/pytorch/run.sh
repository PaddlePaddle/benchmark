#!/bin/bash

cmd="python3 $1.py --dataroot ./datasets/horse2zebra --name horse2zebra_cyclegan --model cycle_gan"

if [ $1 == "train" ]; then
  cmd=${cmd}' --display_id -1'
fi

echo $#

if [ $# == 2 ]; then
  cmd=${cmd}" --gpu_ids $2"
fi


echo $cmd

${cmd}
