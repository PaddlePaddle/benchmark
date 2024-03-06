#!/usr/bin/env bash

pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple

echo "*******prepare benchmark start ***********"

pip install -U pip
echo `pip --version`

pip install torch==1.2.0 torchvision==0.17.0
pip install lmdb pillow torchvision nltk natsort

rm -rf ./datasets/ocr_rec_dataset_examples/
wget -nc -P ./datasets/ https://paddle-model-ecology.bj.bcebos.com/uapi/data/ocr_rec_dataset_examples.tar --no-check-certificate
cd ./datasets/ && tar xf ocr_rec_dataset_examples.tar
cd ./ocr_rec_dataset_examples/
for i in `seq 10`; do cp train.txt "dup${i}".txt; done
cat dup* > train.txt && rm -rf dup*
cd ../
python create_lmdb_dataset.py --inputPath ./ocr_rec_dataset_examples/ --gtFile ./ocr_rec_dataset_examples/train.txt --outputPath ./ocr_rec_dataset_examples/train_data_lmdb/
python create_lmdb_dataset.py --inputPath ./ocr_rec_dataset_examples/ --gtFile ./ocr_rec_dataset_examples/val.txt --outputPath ./ocr_rec_dataset_examples/val_data_lmdb/
cd ../

echo "*******prepare benchmark end***********"
