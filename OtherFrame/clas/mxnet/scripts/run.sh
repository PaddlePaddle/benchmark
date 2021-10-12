#!/usr/bin/env bash
sed -i '{s/batch\[j\]/batch/g}' scripts/classification/imagenet/train_imagenet.py

train_cmd_sp="python scripts/classification/imagenet/train_imagenet.py --data-dir ./data --model mobilenet1.0 --batch-size 64 --mode hybrid --num-gpus 1 2>&1|tee sp.log"
train_cmd_mp="python scripts/classification/imagenet/train_imagenet.py --data-dir ./data --model mobilenet1.0 --mode hybrid --batch-size 64 --num-gpus 8 -j 32 2>&1|tee mp.log"

mkdir logs
eval ${train_cmd_sp}
python benchmark/analysis_log.py -f sp.log -m MobileNetV1 -b 64 -n 1
sleep 20

eval ${train_cmd_mp}
python benchmark/analysis_log.py -f mp.log -m MobileNetV1 -b 64 -n 8
mv clas* logs/
