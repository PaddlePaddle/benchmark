#!/usr/bin/env bash

pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple

echo "*******prepare benchmark start ***********"

pip install -U pip
echo `pip --version`

#pip install torch==2.2.0 torchvision==0.17.0
# install env
wget -c --no-proxy ${FLAG_TORCH_WHL_URL}
tar_file_name=$(echo ${FLAG_TORCH_WHL_URL} | awk -F '/' '{print $NF}')
dir_name=$(echo ${tar_file_name} | awk -F '.' '{print $1}')
tar xf ${tar_file_name}
rm -rf ${tar_file_name}

pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
pip install ${dir_name}/*

pip install --upgrade setuptools  # 避免安装 fire 失败
pip install lmdb pillow nltk natsort fire

rm -rf ./datasets/ocr_rec_dataset_examples/
wget -nc -P ./datasets/ https://paddle-model-ecology.bj.bcebos.com/uapi/data/ocr_rec_dataset_examples.tar --no-check-certificate
cd ./datasets/ && tar xf ocr_rec_dataset_examples.tar
cd ./ocr_rec_dataset_examples/
for i in `seq 20`; do cp train.txt "dup${i}".txt; done
cat dup* > train.txt && rm -rf dup*
echo `head -n 1 val.txt` > val.txt
sed -i $'s/ /\t/g' val.txt
cd ../../
python create_lmdb_dataset.py --inputPath ./datasets/ocr_rec_dataset_examples/ --gtFile ./datasets/ocr_rec_dataset_examples/train.txt --outputPath ./datasets/ocr_rec_dataset_examples/train_data_lmdb/
python create_lmdb_dataset.py --inputPath ./datasets/ocr_rec_dataset_examples/ --gtFile ./datasets/ocr_rec_dataset_examples/val.txt --outputPath ./datasets/ocr_rec_dataset_examples/val_data_lmdb/

echo "*******prepare benchmark end***********"
