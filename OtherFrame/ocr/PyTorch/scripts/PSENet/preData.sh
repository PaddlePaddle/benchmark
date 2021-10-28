# 下载数据集并解压
cd /workspace/models/PSENet

wget -c  -p ./tain_data/  https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/icdar2015.tar && cd train_data  && tar xf icdar2015.tar && cd ../