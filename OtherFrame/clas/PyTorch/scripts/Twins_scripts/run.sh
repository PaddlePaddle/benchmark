#!/usr/bin/env bash

train_cmd_mp="python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model alt_gvt_base --batch-size 128 --data-path data/imagenet --dist-eval --drop-path 0.3"

train_cmd_sp="python main.py --model alt_gvt_base --batch-size 128 --data-path data/imagenet --dist-eval --drop-path 0.3"

rm -rf log_*
timeout 10m ${train_cmd_mp} 2>&1|tee log_mp.txt
python analysis_log.py -f log_mp.txt -m alt_gvt_base -b 128 -n 8
sleep 20

timeout 10m ${train_cmd_sp} 2>&1|tee log_sp.txt
python analysis_log.py -f log_sp.txt -m alt_gvt_base -b 128 -n 1
