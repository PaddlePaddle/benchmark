# 下载数据集并解压
cd /workspace/models/PSENet
rm -rf train_data
wget -P ./train_data/ -N https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/icdar2015.tar && cd train_data  && tar xf icdar2015.tar && cd ../
