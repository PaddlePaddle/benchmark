#!/usr/bin/env bash
# 在TimeSformer目录下执行
echo "*******prepare benchmark start ***********"
################################# 安装最新版pip
#pip install -U pip
echo `pip --version`

echo "https_proxy $HTTPS_PRO" 
echo "http_proxy $HTTP_PRO" 
export https_proxy=$HTTPS_PRO
export http_proxy=$HTTP_PRO
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com

################################# 安装TimeSformer的环境依赖
pip install -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir
pip install -U simplejson -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U traitlets -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U importlib-metadata -i https://pypi.tuna.tsinghua.edu.cn/simple


################################# 安装torch 1.8.1和torchvision 0.9.1
#pip install torch-1.9.1+cu111 -i https://pypi.tuna.tsinghua.edu.cn/simple --force-reinstall
#pip install torchvision-0.10.1+cu111 -i https://pypi.tuna.tsinghua.edu.cn/simple --force-reinstall
wget https://paddle-wheel.bj.bcebos.com/benchmark/torch-1.8.1%2Bcu111-cp37-cp37m-linux_x86_64.whl
wget https://paddle-wheel.bj.bcebos.com/benchmark/torchvision-0.9.1%2Bcu111-cp37-cp37m-linux_x86_64.whl
pip install torch-1.8.1+cu111-cp37-cp37m-linux_x86_64.whl
pip install torchvision-0.9.1+cu111-cp37-cp37m-linux_x86_64.whl

################################# 以包的形式安装TimeSformer(使用pip安装)
python setup.py build
pip install ./

################################# 下载预训练模型到目录下
wget -nc https://videotag.bj.bcebos.com/PaddleVideo-release2.2/jx_vit_base_p16_224-80ecf9dd.pth

cp run_net.py tools/run_net.py
################################# 准备训练数据
#wget -nc https://videotag.bj.bcebos.com/Data/k400_videos_small.tar
#tar -xf k400_videos_small.tar
wget -nc https://paddle-wheel.bj.bcebos.com/benchmark/train_small_videos.list
wget -nc https://paddle-wheel.bj.bcebos.com/benchmark/val_small_videos.list
wget -nc https://videotag.bj.bcebos.com/Data/K400_fleet_benchmark/part1.tar
wget -nc https://videotag.bj.bcebos.com/Data/K400_fleet_benchmark/part2.tar
wget -nc https://videotag.bj.bcebos.com/Data/K400_fleet_benchmark/part3.tar
rm -rf videos/
mv train_small_videos.list annotations/train.txt
mv val_small_videos.list annotations/val.txt
tar -zxvf part1.tar
tar -zxvf part2.tar
tar -zxvf part3.tar
mkdir videos
mv part1/* part2/* part3/* -t videos/
rm -rf part*
echo "*******prepare benchmark end***********"
