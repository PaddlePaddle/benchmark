#!/usr/bin/env bash

# install env
rm -rf torch*
wget -c https://paddle-imagenet-models-name.bj.bcebos.com/benchmark/torch-1.7.1%2Bcu110-cp37-cp37m-linux_x86_64.whl
wget -c https://paddle-imagenet-models-name.bj.bcebos.com/benchmark/torchvision-0.8.2%2Bcu110-cp37-cp37m-linux_x86_64.whl

pip install matplotlib easydict opencv-python einops pyyaml
pip install scikit-image imgaug PyTurboJPEG
pip install scikit-learn
pip install termcolor imgaug prettytable
pip install torch-1.7.1+cu110-cp37-cp37m-linux_x86_64.whl torchvision-0.8.2+cu110-cp37-cp37m-linux_x86_64.whl
pip install timm==0.3.4

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
