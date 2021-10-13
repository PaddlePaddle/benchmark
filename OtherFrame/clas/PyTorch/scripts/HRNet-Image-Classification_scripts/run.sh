#!/usr/bin/env bash

sed -i '{s/view/reshape/g}' lib/core/evaluate.py
sed -i '{s/PRINT_FREQ: 1000/PRINT_FREQ: 10/g}' experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml

train_cmd="python tools/train.py --cfg experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml"

rm -rf output
sed -ie 's/GPUS: (0,1,2,3)/GPUS: (0,)/g' experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
timeout 15m ${train_cmd}
python analysis_log.py -d output -m HRNet_W48_C -b 32 -n 1
sleep 20
sed -ie 's/GPUS: (0,)/GPUS: (0,1,2,3,4,5,6,7)/g' experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
rm -rf output
sed -i '{s/WORKERS: 4/WORKERS: 32/g}' experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
timeout 15m ${train_cmd}
python analysis_log.py -d output -m HRNet_W48_C -b 32 -n 8
sed -ie 's/GPUS: (0,1,2,3,4,5,6,7)/GPUS: (0,1,2,3)/g' experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
sed -i '{s/WORKERS: 32/WORKERS: 4/g}' experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
