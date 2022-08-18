#!/usr/bin/env bash
# 执行路径在模型库的根目录下
################################# 安装框架 如:
echo "*******prepare benchmark start ***********"
#pip install -U pip
echo `pip --version`

echo "https_proxy $HTTPS_PRO" 
echo "http_proxy $HTTP_PRO" 
export https_proxy=$HTTPS_PRO
export http_proxy=$HTTP_PRO
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com

pip install setuptools==50.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install Cython -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
#pip install torch==1.10.0 torchvision==0.11.1
wget https://paddle-wheel.bj.bcebos.com/benchmark/torch-1.10.0%2Bcu111-cp37-cp37m-linux_x86_64.whl -O torch-1.10.0+cu111-cp37-cp37m-linux_x86_64.whl
wget https://paddle-wheel.bj.bcebos.com/benchmark/torchvision-0.11.1%2Bcu111-cp37-cp37m-linux_x86_64.whl -O torchvision-0.11.1+cu111-cp37-cp37m-linux_x86_64.whl
wget https://paddle-wheel.bj.bcebos.com/benchmark/mmcv_full-1.4.4-cp37-cp37m-linux_x86_64_A100.whl -O mmcv_full-1.4.4-cp37-cp37m-linux_x86_64.whl
pip install torch-1.10.0+cu111-cp37-cp37m-linux_x86_64.whl
pip install torchvision-0.11.1+cu111-cp37-cp37m-linux_x86_64.whl
pip install mmcv_full-1.4.4-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U scikit-learn -i https://pypi.douban.com/simple
pip install -U mccabe -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U pillow -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
pip install -v -e .

wget -P /root/.cache/torch/hub/checkpoints/ https://paddle-wheel.bj.bcebos.com/benchmark/resnet50-0676ba61.pth

cp dist_train.sh tools/dist_train.sh
################################# 准备训练数据 如:
#wget -nc -P data/coco/ https://paddledet.bj.bcebos.com/data/coco_benchmark.tar
wget -nc -P data/coco/ https://bj.bcebos.com/v1/paddledet/data/coco.tar
cd ./data/coco/ && tar -xf coco.tar && mv -u coco/* .
rm -rf coco/ && cd ../../
echo "*******prepare benchmark end***********"
