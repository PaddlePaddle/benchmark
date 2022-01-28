#!/usr/bin/env bash
# 执行路径在模型库的根目录下
################################# 安装框架 如:
echo "*******prepare benchmark start ***********"
pip install -U pip
echo `pip --version`
python ${BENCHMARK_ROOT}/paddlecloud/file_upload_download.py \
    --remote-path frame_benchmark/pytorch_req/pytorch_191/ \
    --local-path ./  \
    --mode download
ls
# pip install torch==1.9.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch-1.9.1-cp37-cp37m-manylinux1_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torchvision==0.10.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install mmcv-full==1.3.13 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e .
pip list

wget https://paddleseg.bj.bcebos.com/benchmark/mmseg/mmseg_benchmark_configs.tar.gz
tar -zxf mmseg_benchmark_configs.tar.gz
################################# 准备训练数据 如:
mkdir -p data
wget https://paddleseg.bj.bcebos.com/dataset/cityscapes_30imgs.tar.gz \
    -O data/cityscapes_30imgs.tar.gz
tar -zxf data/cityscapes_30imgs.tar.gz -C data/
mv data/cityscapes_30imgs data/cityscapes
echo "*******prepare benchmark end***********"




