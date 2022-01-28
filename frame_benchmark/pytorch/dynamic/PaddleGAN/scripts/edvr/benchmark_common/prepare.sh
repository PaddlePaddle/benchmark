#!/usr/bin/env bash

################################# 安装框架 如:
pip install -U pip
echo `pip --version`
pip install torch==1.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchvision==0.10.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html

pip install mmcv-full==1.3.13 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html

pip install -r requirements.txt
pip install -v -e .

################################# 准备训练数据 如:
mkdir -p data/REDS 
python ${BENCHMARK_ROOT}/paddlecloud/file_upload_download.py \
--remote-path frame_benchmark/paddle/PaddleGAN/REDS/test_sharp \
--local-path ./data/REDS \
--mode download
python ${BENCHMARK_ROOT}/paddlecloud/file_upload_download.py \
--remote-path frame_benchmark/paddle/PaddleGAN/REDS/test_sharp_bicubic \
--local-path ./data/REDS \
--mode download
python ${BENCHMARK_ROOT}/paddlecloud/file_upload_download.py \
--remote-path frame_benchmark/paddle/PaddleGAN/REDS/train_sharp \
--local-path ./data/REDS \
--mode download
python ${BENCHMARK_ROOT}/paddlecloud/file_upload_download.py \
--remote-path frame_benchmark/paddle/PaddleGAN/REDS/train_sharp_bicubic \
--local-path ./data/REDS \
--mode download
tar -vxf data/REDS/train_sharp.tar -C data/REDS
tar -vxf data/REDS/train_sharp_bicubic.tar -C data/REDS
tar -vxf data/REDS/REDS4_test_sharp.tar -C data/REDS
tar -vxf data/REDS/REDS4_test_sharp_bicubic.tar -C data/REDS
echo "download data" #waiting data process
echo "dataset prepared done" 

echo "*******prepare benchmark end***********"


