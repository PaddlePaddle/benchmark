#!/usr/bin/env bash
# 执行路径在模型库的根目录下
################################# 安装框架 如:
echo "*******prepare benchmark start ***********"
python3.7 -m pip install -U pip
echo `python3.7 -m pip --version`
# python3.7 ${BENCHMARK_ROOT}/paddlecloud/file_upload_download.py \
#     --remote-path frame_benchmark/pytorch_req/pytorch_191/ \
#     --local-path ./  \
#     --mode download
ls
# python3.7 -m pip install https://download.pytorch.org/whl/cu102/torch-1.8.0-cp37-cp37m-linux_x86_64.whl --no-deps
python3.7 -m pip install -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple --no-deps
python3.7 setup.py build develop
python3.7 -m pip list
 # 下载预训练模型
wget -nc https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth

################################# 准备训练数据:
wget -nc https://videotag.bj.bcebos.com/Data/k400_videos_small.tar
tar -xf k400_videos_small.tar
echo "*******prepare benchmark end***********"