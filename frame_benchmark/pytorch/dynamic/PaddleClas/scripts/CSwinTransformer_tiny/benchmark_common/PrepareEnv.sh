#!/usr/bin/env bash

# install env
wget -c --no-proxy ${FLAG_TORCH_WHL_URL}
tar_file_name=$(echo ${FLAG_TORCH_WHL_URL} | awk -F '/' '{print $NF}')
dir_name=$(echo ${tar_file_name} | awk -F '.' '{print $1}')
tar xf ${tar_file_name}
rm -rf ${tar_file_name}

pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
pip install ${dir_name}/*
pip install -r requirements.txt
# fix bug to adapt to torch 2.0
sed -i 's/local_rank/local-rank/g' train.py

# # install env
# rm -rf torch*
# wget -c https://paddle-imagenet-models-name.bj.bcebos.com/benchmark/torch-1.7.1%2Bcu110-cp37-cp37m-linux_x86_64.whl
# wget -c https://paddle-imagenet-models-name.bj.bcebos.com/benchmark/torchvision-0.8.2%2Bcu110-cp37-cp37m-linux_x86_64.whl

# pip install matplotlib easydict opencv-python einops pyyaml
# pip install scikit-image imgaug PyTurboJPEG
# pip install scikit-learn
# pip install termcolor imgaug prettytable
# pip install torch-1.7.1+cu110-cp37-cp37m-linux_x86_64.whl torchvision-0.8.2+cu110-cp37-cp37m-linux_x86_64.whl
# pip install timm==0.3.4

dataset_url="https://paddle-imagenet-models-name.bj.bcebos.com/data/ImageNet1k/ILSVRC2012_val.tar"

# prepare data
rm -rf dataset/ILSVRC2012_val.tar
rm -rf dataset/ILSVRC2012_val
rm -rf dataset/ILSVRC2012_name_train.txt
rm -rf dataset/ILSVRC2012_name_val.txt

cd dataset
wget -c ${dataset_url}
tar xf ILSVRC2012_val.tar
ln -s ILSVRC2012_val/ILSVRC2012_val_dirs train
ln -s ILSVRC2012_val/ILSVRC2012_val_dirs val
sed -ri 's/ILSVRC2012_val_dirs\///' ILSVRC2012_val/val_list.txt
sed -ri 's/ .*//' ILSVRC2012_val/val_list.txt
rm -rf ILSVRC2012_name_train.txt ILSVRC2012_name_val.txt
ln -s ILSVRC2012_val/val_list.txt ILSVRC2012_name_train.txt
ln -s ILSVRC2012_val/val_list.txt ILSVRC2012_name_val.txt
cd ..

echo "*******prepare benchmark end***********"
