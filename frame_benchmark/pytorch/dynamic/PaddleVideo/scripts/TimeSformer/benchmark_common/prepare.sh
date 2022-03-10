#!/usr/bin/env bash
# 在TimeSformer目录下执行
echo "*******prepare benchmark start ***********"
################################# 安装最新版pip
pip install -U pip
echo `pip --version`

################################# 安装TimeSformer的环境依赖
pip install -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir
pip install simplejson

################################# 安装torch 1.8.1
pip install torch==1.8.1 -i https://pypi.tuna.tsinghua.edu.cn/simple --force-reinstall

################################# 以包的形式安装TimeSformer(使用pip安装)
python setup.py build
pip install ./

################################# 下载预训练模型到目录下
wget -nc https://videotag.bj.bcebos.com/PaddleVideo-release2.2/jx_vit_base_p16_224-80ecf9dd.pth

################################# 准备训练数据
wget -nc https://videotag.bj.bcebos.com/Data/k400_videos_small.tar
tar -xf k400_videos_small.tar
echo "*******prepare benchmark end***********"